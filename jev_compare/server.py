"""Local comparison bench: run labelled classification tasks through several
models and stream per-item results (answers, confidence, latency, tokens, cost)
to the browser."""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from jev_compare.providers import load_providers, options

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Point these at your own folders to run private task sets or keep runs elsewhere.
TASKS_DIR = Path(os.getenv("JEV_COMPARE_TASKS", ROOT / "tasks"))
RUNS_DIR = Path(os.getenv("JEV_COMPARE_RUNS", ROOT / "runs"))
RUNS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="jev-compare")


def load_tasks() -> dict[str, dict]:
    tasks = {}
    for p in sorted(TASKS_DIR.glob("*.json")):
        t = json.loads(p.read_text())
        # Context every item shares (e.g. a policy) is sent with each item, as it would be in production.
        if shared := t.get("shared_state"):
            for it in t["items"]:
                it["state"] = {**shared, **it["state"]}
        tasks[t["id"]] = t
    return tasks


def load_models() -> list[dict]:
    return json.loads((ROOT / "config" / "models.json").read_text())["models"]


def norm_label(q: dict, v) -> str:
    if q["type"] == "noul":
        return "true" if v in (True, "true", "True", 1) else "false"
    return str(v)


def grade(result: dict, qid: str, labels: dict) -> bool | None:
    """True or False against the known answer; None when the item has no known answer for this question."""
    if qid not in labels:
        return None
    return result["answers"].get(qid, {}).get("answer") == labels[qid]


def preview(item: dict) -> str:
    s = item["state"]
    if isinstance(s, dict):
        main = next((s[k] for k in ("text", "body", "article", "summary") if k in s), None)
        if main is None and "ingredients" in s:
            main = f"{s.get('name', '')}: {', '.join(s['ingredients'])}"
        lead = s.get("subject") or s.get("title")
        s = f"{lead}. {main}" if lead and main else (main or json.dumps(s))
    return " ".join(str(s).split())[:160]


@app.get("/")
def single_call_page():
    return FileResponse(ROOT / "static" / "call.html")


@app.get("/bench")
def bench_page():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/preview")
def preview_requests(task_id: str, item_id: str):
    """The exact request each model would receive for this item. Built locally; nothing is sent."""
    task = load_tasks().get(task_id)
    item = next((i for i in task["items"] if i["id"] == item_id), None) if task else None
    if item is None:
        raise HTTPException(404, "unknown task or item")
    out = []
    for pid, prov in load_providers(load_models()).items():
        url, body = getattr(prov, "real", prov).build_request(task, item)
        out.append({"model": pid, "method": "POST", "url": url, "body": body})
    return out


class CallRequest(BaseModel):
    task_id: str
    item_id: str


@app.post("/api/call")
async def single_call(req: CallRequest):
    """Send one item to every model at once, one request each (no retries), and return
    what was sent, what came back, and the parsed answers against the known answers."""
    tasks = load_tasks()
    if req.task_id not in tasks:
        raise HTTPException(404, "unknown task")
    task = tasks[req.task_id]
    item = next((i for i in task["items"] if i["id"] == req.item_id), None)
    if item is None:
        raise HTTPException(404, "unknown item")
    qs = {q["id"]: q for q in task["questions"]}
    labels = {k: norm_label(qs[k], v) for k, v in item.get("labels", {}).items()}
    provs = load_providers(load_models())
    for p in provs.values():
        p.retries = False

    async with httpx.AsyncClient(timeout=60) as client:

        async def one(pid):
            r = (await provs[pid].classify(client, task, item)).to_dict()
            r["correct"] = {qid: grade(r, qid, labels) for qid in qs}
            return r

        results = await asyncio.gather(*(one(pid) for pid in provs))
    return {
        "task_id": task["id"],
        "item": {"id": item["id"], "state": item["state"], "labels": labels},
        "results": results,
    }


@app.get("/api/config")
def config():
    tasks = load_tasks()
    provs = load_providers(load_models())
    return {
        "tasks": [
            {
                "id": t["id"],
                "title": t["title"],
                "use_case": t.get("use_case"),
                "description": t["description"],
                "volume": t.get("annual_volume", 100000),
                "count": len(t["items"]),
                "items": [{"id": it["id"], "preview": preview(it)} for it in t["items"]],
                "questions": [{**q, "options": options(q)} for q in t["questions"]],
            }
            for t in tasks.values()
        ],
        "models": [
            {
                **m,
                "live": not getattr(provs[m["id"]], "real", None),
                "model_name": getattr(getattr(provs[m["id"]], "real", provs[m["id"]]), "model", m["id"]),
            }
            for m in load_models()
        ],
    }


class RunRequest(BaseModel):
    task_id: str
    models: list[str]
    n: int = 30
    concurrency: int = 6
    simulate: bool = False


@app.post("/api/run")
async def run(req: RunRequest):
    tasks = load_tasks()
    if req.task_id not in tasks:
        raise HTTPException(404, "unknown task")
    task = tasks[req.task_id]
    cfg = [m for m in load_models() if m["id"] in req.models]
    if not cfg:
        raise HTTPException(400, "pick at least one model")
    provs = load_providers(cfg, force_sim=req.simulate)

    items = list(task["items"])
    random.Random(7).shuffle(items)
    items = items[: max(1, min(req.n, len(items)))]
    qs = {q["id"]: q for q in task["questions"]}
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{task['id']}"

    async def stream():
        results: list[dict] = []
        queue: asyncio.Queue = asyncio.Queue()
        start = {
            "type": "start",
            "run_id": run_id,
            "task_id": task["id"],
            "models": [{"id": m["id"], "simulated": hasattr(provs[m["id"]], "real")} for m in cfg],
            "items": [
                {
                    "id": it["id"],
                    "preview": preview(it),
                    "state": it["state"],
                    "labels": {k: norm_label(qs[k], v) for k, v in it.get("labels", {}).items()},
                }
                for it in items
            ],
        }
        yield json.dumps(start) + "\n"

        async with httpx.AsyncClient(timeout=60) as client:

            async def worker(pid: str, sem: asyncio.Semaphore, item: dict):
                async with sem:
                    try:
                        r = (await provs[pid].classify(client, task, item)).to_dict()
                    except Exception as e:
                        r = {
                            "model": pid,
                            "item_id": item["id"],
                            "ok": False,
                            "error": repr(e),
                            "answers": {},
                            "latency_ms": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "cost_usd": 0,
                            "simulated": False,
                            "raw": None,
                        }
                    labels = {k: norm_label(qs[k], v) for k, v in item.get("labels", {}).items()}
                    r["correct"] = {qid: grade(r, qid, labels) for qid in qs}
                    r["finished_at"] = time.time()
                    await queue.put(r)

            jobs = []
            for pid in provs:
                sem = asyncio.Semaphore(max(1, req.concurrency))
                jobs += [asyncio.create_task(worker(pid, sem, it)) for it in items]

            for _ in range(len(jobs)):
                r = await queue.get()
                results.append(r)
                yield json.dumps({"type": "result", **r}, default=str) + "\n"

        record = {**start, "type": "run", "created": datetime.now().isoformat(), "results": results}
        (RUNS_DIR / f"{run_id}.json").write_text(json.dumps(record, default=str))
        yield json.dumps({"type": "done", "run_id": run_id}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/api/runs")
def runs():
    out = []
    for p in sorted(RUNS_DIR.glob("*.json"), reverse=True)[:30]:
        d = json.loads(p.read_text())
        out.append(
            {
                "run_id": d["run_id"],
                "task_id": d["task_id"],
                "created": d["created"],
                "models": d["models"],
                "n": len(d["items"]),
            }
        )
    return out


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    p = RUNS_DIR / f"{Path(run_id).name}.json"
    if not p.exists():
        raise HTTPException(404, "run not found")
    return json.loads(p.read_text())
