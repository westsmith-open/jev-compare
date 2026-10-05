"""Model adapters.

Every adapter takes the same task question set and one item, and returns a
normalised result: one answer per question with a confidence (and the full
probability distribution where the model gives one), plus latency, tokens and
cost. That is what lets the UI compare a decision model with a general LLM on
equal terms.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Question helpers


def options(q: dict) -> list[str]:
    """Allowed answer keys for a question, in order."""
    if q["type"] == "noul":
        return ["true", "false"]
    if q["type"] == "choice":
        return list(q["criteria"].keys())
    if q["type"] == "score":
        return [lvl["key"] for lvl in q["levels"]]
    raise ValueError(f"unknown question type {q['type']}")


def state_text(item: dict) -> str:
    s = item["state"]
    return s if isinstance(s, str) else json.dumps(s, indent=2, ensure_ascii=False)


@dataclass
class Result:
    model: str
    item_id: str
    simulated: bool = False
    ok: bool = True
    error: str | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    answers: dict[str, dict] = field(default_factory=dict)
    raw: Any = None
    request: Any = None  # exactly what was sent, minus the auth header

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def answer_from_probs(q: dict, probs: dict[str, float]) -> dict:
    opts = options(q)
    probs = {k: float(probs.get(k, 0.0)) for k in opts}
    total = sum(probs.values()) or 1.0
    probs = {k: v / total for k, v in probs.items()}
    best = max(opts, key=lambda k: probs[k])
    return {"answer": best, "confidence": probs[best], "probs": probs}


# ---------------------------------------------------------------------------
# Base


class Provider:
    retries = True  # single-call view turns this off so timings are one clean request

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.id = cfg["id"]

    @property
    def has_key(self) -> bool:
        return False

    def cost(self, tin: int, tout: int) -> float:
        return (tin * self.cfg["price_in"] + tout * self.cfg["price_out"]) / 1e6

    async def classify(self, client: httpx.AsyncClient, task: dict, item: dict) -> Result:
        raise NotImplementedError


async def post_with_retry(client: httpx.AsyncClient, url: str, attempts: int = 3, **kw) -> httpx.Response:
    for attempt in range(attempts):
        resp = await client.post(url, **kw)
        if resp.status_code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
            return resp
        await asyncio.sleep(0.5 * 2**attempt)
    return resp


# ---------------------------------------------------------------------------
# TypeSafe Jev (System One API)
#
# Request: {"model", "state", "questions": {id: {type, instructions, criteria}}}
#   choice criteria = {key: description}; score criteria = [level text, low→high];
#   noul criteria = {"true": ..., "false": ...}
# Response: {"answers": {id: {...}}, "usage": {...}}. The parser below accepts
# the field-name variants seen across the TypeSafe, OpenRouter and aimlapi docs,
# and always prefers the probability distribution when one is present.


class JevProvider(Provider):
    """Direct to TypeSafe when TYPESAFE_API_KEY is set, otherwise via OpenRouter's Decisions API."""

    @property
    def direct(self) -> bool:
        return bool(os.getenv("TYPESAFE_API_KEY"))

    @property
    def has_key(self) -> bool:
        return self.direct or bool(os.getenv("OPENROUTER_API_KEY"))

    @property
    def key(self) -> str:
        return os.environ["TYPESAFE_API_KEY"] if self.direct else os.environ["OPENROUTER_API_KEY"]

    @property
    def url(self) -> str:
        default = "https://api.typesafe.ai/v1/systemone" if self.direct else "https://openrouter.ai/api/alpha/decisions"
        return os.getenv("JEV_URL", default)

    @property
    def model(self) -> str:
        return os.getenv("JEV_MODEL", "jev-1.13" if self.direct else "typesafe/jev-1.13")

    def build_questions(self, task: dict) -> dict:
        out = {}
        for q in task["questions"]:
            if q["type"] == "score":
                crit: Any = [lvl["text"] for lvl in q["levels"]]
            else:
                crit = q["criteria"]
            out[q["id"]] = {"type": q["type"], "instructions": q["instructions"], "criteria": crit}
        return out

    def parse_answer(self, q: dict, a: Any) -> dict:
        opts = options(q)
        if q["type"] == "noul":
            # OpenRouter Decisions API: {"type": "noul", "noul": <P(true)>}
            if isinstance(a, (int, float)):
                p = float(a)
            else:
                key = next((k for k in ("noul", "probability", "p", "value") if k in a), None)
                p = float(a[key] if key else a["probabilities"]["true"])
            return answer_from_probs(q, {"true": p, "false": 1 - p})

        probs = a.get("probabilities") if isinstance(a, dict) else None
        if isinstance(probs, list):
            return answer_from_probs(q, dict(zip(opts, probs, strict=False)))
        if isinstance(probs, dict):
            mapped: dict[str, float] = {}
            texts = [lvl["text"] for lvl in q.get("levels", [])]
            for k, v in probs.items():
                if k in opts:
                    mapped[k] = v
                elif k in texts:
                    mapped[opts[texts.index(k)]] = v
                elif str(k).isdigit() and int(k) < len(opts):
                    mapped[opts[int(k)]] = v
            if mapped:
                return answer_from_probs(q, mapped)

        # No distribution: fall back to the selected value.
        sel = None
        if isinstance(a, dict):
            for key in ("choice", "answer", "selected", "value", "level", "index"):
                if key in a:
                    sel = a[key]
                    break
        else:
            sel = a
        if isinstance(sel, (int, float)) and q["type"] == "score":
            sel = opts[max(0, min(len(opts) - 1, round(sel)))]
        conf = float(a.get("confidence", 0.0)) if isinstance(a, dict) else 0.0
        return {"answer": str(sel), "confidence": conf, "probs": None}

    def build_request(self, task: dict, item: dict) -> tuple[str, dict]:
        return self.url, {"model": self.model, "state": item["state"], "questions": self.build_questions(task)}

    async def classify(self, client, task, item) -> Result:
        url, body = self.build_request(task, item)
        headers = {"Authorization": f"Bearer {self.key}"}
        t0 = time.perf_counter()
        resp = await post_with_retry(client, url, 3 if self.retries else 1, json=body, headers=headers)
        latency = (time.perf_counter() - t0) * 1000
        r = Result(
            model=self.id, item_id=item["id"], latency_ms=latency, request={"method": "POST", "url": url, "body": body}
        )
        if resp.status_code != 200:
            r.ok, r.error, r.raw = False, f"HTTP {resp.status_code}: {resp.text[:300]}", None
            return r
        data = resp.json()
        r.raw = data
        answers = data.get("answers", data)
        for q in task["questions"]:
            try:
                r.answers[q["id"]] = self.parse_answer(q, answers[q["id"]])
            except Exception as e:  # keep the run going; surface in the audit view
                r.answers[q["id"]] = {"answer": None, "confidence": 0.0, "probs": None, "error": str(e)}
        usage = data.get("usage", {})
        r.input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)))
        r.output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)))
        r.cost_usd = float(usage["cost"]) if "cost" in usage else self.cost(r.input_tokens, r.output_tokens)
        return r


# ---------------------------------------------------------------------------
# Google Gemini (generateContent with a JSON response schema)
#
# A general LLM has no calibrated distribution, so we do what most teams do
# today: constrain the answer to the allowed values and ask for a self-reported
# confidence. The comparison shows how far that confidence can be trusted.


class GeminiProvider(Provider):
    """Direct to Google when GEMINI_API_KEY is set, otherwise via OpenRouter chat completions."""

    @property
    def direct(self) -> bool:
        return bool(os.getenv("GEMINI_API_KEY"))

    @property
    def has_key(self) -> bool:
        return self.direct or bool(os.getenv("OPENROUTER_API_KEY"))

    @property
    def model(self) -> str:
        return os.getenv("GEMINI_MODEL", "gemini-3.8-flash" if self.direct else "google/gemini-3.8-flash")

    def prompt(self, task: dict, item: dict) -> str:
        kinds = {"noul": "yes/no", "choice": "pick one", "score": "ordered scale, lowest first"}
        lines = [
            "## Task",
            "",
            task["description"],
            "",
            "## Input",
            "",
            "```json",
            state_text(item),
            "```",
            "",
            "## Questions",
            "",
            "Answer every question using only the allowed keys. "
            "Give `confidence` as the probability (0 to 1) that your answer is correct.",
        ]
        for q in task["questions"]:
            lines += ["", f"### `{q['id']}` ({kinds[q['type']]})", "", q["instructions"], ""]
            if q["type"] == "noul":
                lines += [f"- `true`: {q['criteria']['true']}", f"- `false`: {q['criteria']['false']}"]
            elif q["type"] == "choice":
                lines += [f"- `{k}`: {v}" for k, v in q["criteria"].items()]
            else:
                lines += [f"- `{lvl['key']}`: {lvl['text']}" for lvl in q["levels"]]
        return "\n".join(lines)

    def schema(self, task: dict) -> dict:
        props = {}
        for q in task["questions"]:
            props[q["id"]] = {
                "type": "OBJECT",
                "properties": {
                    "answer": {"type": "STRING", "enum": options(q)},
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["answer", "confidence"],
                "propertyOrdering": ["answer", "confidence"],
            }
        ids = [q["id"] for q in task["questions"]]
        return {"type": "OBJECT", "properties": props, "required": ids, "propertyOrdering": ids}

    def parse_answers(self, r: Result, task: dict, text: str) -> None:
        parsed = json.loads(text)
        for q in task["questions"]:
            a = parsed.get(q["id"], {})
            conf = float(a.get("confidence", 0.0))
            conf = max(0.0, min(1.0, conf / 100 if conf > 1 else conf))
            r.answers[q["id"]] = {"answer": a.get("answer"), "confidence": conf, "probs": None}

    async def classify(self, client, task, item) -> Result:
        return await (self.classify_direct if self.direct else self.classify_openrouter)(client, task, item)

    def build_request(self, task: dict, item: dict) -> tuple[str, dict]:
        return (self.build_request_direct if self.direct else self.build_request_openrouter)(task, item)

    def build_request_openrouter(self, task: dict, item: dict) -> tuple[str, dict]:
        def strict(schema: dict) -> dict:  # Gemini-style schema → JSON Schema with strict objects
            out = {k: v for k, v in schema.items() if k != "propertyOrdering"}
            out["type"] = schema["type"].lower()
            if "properties" in schema:
                out["properties"] = {k: strict(v) for k, v in schema["properties"].items()}
                out["additionalProperties"] = False
            return out

        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a precise classifier. Reply with JSON only."},
                {"role": "user", "content": self.prompt(task, item)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answers", "strict": True, "schema": strict(self.schema(task))},
            },
            "temperature": 0,
            "usage": {"include": True},
        }
        if effort := self.cfg.get("reasoning_effort") or os.getenv("GEMINI_REASONING_EFFORT"):
            body["reasoning"] = {"effort": effort}
        return "https://openrouter.ai/api/v1/chat/completions", body

    async def classify_openrouter(self, client, task, item) -> Result:
        url, body = self.build_request_openrouter(task, item)
        headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
        t0 = time.perf_counter()
        resp = await post_with_retry(client, url, 3 if self.retries else 1, json=body, headers=headers)
        latency = (time.perf_counter() - t0) * 1000
        r = Result(
            model=self.id, item_id=item["id"], latency_ms=latency, request={"method": "POST", "url": url, "body": body}
        )
        if resp.status_code != 200:
            r.ok, r.error = False, f"HTTP {resp.status_code}: {resp.text[:300]}"
            return r
        data = resp.json()
        r.raw = data
        usage = data.get("usage", {})
        r.input_tokens = int(usage.get("prompt_tokens", 0))
        r.output_tokens = int(usage.get("completion_tokens", 0))  # includes reasoning tokens
        r.cost_usd = float(usage["cost"]) if "cost" in usage else self.cost(r.input_tokens, r.output_tokens)
        try:
            self.parse_answers(r, task, data["choices"][0]["message"]["content"])
        except Exception as e:
            r.ok, r.error = False, f"Unparseable response: {e}"
        return r

    def build_request_direct(self, task: dict, item: dict) -> tuple[str, dict]:
        gen: dict[str, Any] = {
            "responseMimeType": "application/json",
            "responseSchema": self.schema(task),
            "temperature": 0,
        }
        if lvl := self.cfg.get("reasoning_effort") or os.getenv("GEMINI_THINKING_LEVEL"):
            gen["thinkingConfig"] = {"thinkingLevel": lvl}
        body = {
            "systemInstruction": {"parts": [{"text": "You are a precise classifier. Reply with JSON only."}]},
            "contents": [{"role": "user", "parts": [{"text": self.prompt(task, item)}]}],
            "generationConfig": gen,
        }
        return f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent", body

    async def classify_direct(self, client, task, item) -> Result:
        url, body = self.build_request_direct(task, item)
        headers = {"x-goog-api-key": os.environ["GEMINI_API_KEY"]}
        t0 = time.perf_counter()
        resp = await post_with_retry(client, url, 3 if self.retries else 1, json=body, headers=headers)
        latency = (time.perf_counter() - t0) * 1000
        r = Result(
            model=self.id, item_id=item["id"], latency_ms=latency, request={"method": "POST", "url": url, "body": body}
        )
        if resp.status_code != 200:
            r.ok, r.error = False, f"HTTP {resp.status_code}: {resp.text[:300]}"
            return r
        data = resp.json()
        r.raw = data
        usage = data.get("usageMetadata", {})
        r.input_tokens = int(usage.get("promptTokenCount", 0))
        r.output_tokens = int(usage.get("candidatesTokenCount", 0)) + int(usage.get("thoughtsTokenCount", 0))
        r.cost_usd = self.cost(r.input_tokens, r.output_tokens)
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            self.parse_answers(r, task, text)
        except Exception as e:
            r.ok, r.error = False, f"Unparseable response: {e}"
        return r


# ---------------------------------------------------------------------------
# Simulated responses, used when a key is missing so the demo still runs.
# Profiles are deliberately plain: the decision model is calibrated and fast,
# the LLM is slower and overconfident. Results are flagged `simulated` and the
# UI says so on every view.

SIM_PROFILES = {
    "jev": {"acc": 0.93, "calibrated": True, "lat_ms": 230, "lat_sd": 0.25, "out_tokens": 0},
    "gemini": {"acc": 0.89, "calibrated": False, "lat_ms": 950, "lat_sd": 0.35, "out_tokens": 90},
}


class SimulatedProvider(Provider):
    def __init__(self, cfg: dict, real: Provider):
        super().__init__(cfg)
        self.real = real
        self.profile = SIM_PROFILES.get(cfg["provider"], SIM_PROFILES["gemini"])

    async def classify(self, client, task, item) -> Result:
        seed = int(hashlib.sha256(f"{self.id}:{task['id']}:{item['id']}".encode()).hexdigest()[:12], 16)
        rng = random.Random(seed)
        p = self.profile
        latency = p["lat_ms"] * math.exp(rng.gauss(0, p["lat_sd"]))
        await asyncio.sleep(latency / 1000)
        r = Result(model=self.id, item_id=item["id"], simulated=True, latency_ms=latency)
        for q in task["questions"]:
            opts = options(q)
            truth = str(item["labels"][q["id"]]).lower() if q["type"] == "noul" else item["labels"][q["id"]]
            if p["calibrated"]:
                conf = min(0.995, max(1 / len(opts) + 0.05, rng.betavariate(9, 1.1)))
                correct = rng.random() < conf
            else:
                correct = rng.random() < p["acc"]
                conf = rng.uniform(0.88, 0.99)
            if correct:
                ans = truth
            else:
                others = [o for o in opts if o != truth]
                if q["type"] == "score":  # near misses on ordered scales
                    i = opts.index(truth)
                    others = [opts[j] for j in (i - 1, i + 1) if 0 <= j < len(opts)]
                ans = rng.choice(others)
            if p["calibrated"]:
                rest = [o for o in opts if o != ans]
                weights = [rng.random() for _ in rest]
                s = sum(weights) or 1
                probs = {ans: conf, **{o: (1 - conf) * w / s for o, w in zip(rest, weights, strict=True)}}
                r.answers[q["id"]] = answer_from_probs(q, probs)
            else:
                r.answers[q["id"]] = {"answer": ans, "confidence": conf, "probs": None}
        r.input_tokens = int(len(state_text(item)) / 4 + 60 * len(task["questions"]))
        r.output_tokens = p["out_tokens"] + (15 * len(task["questions"]) if p["out_tokens"] else 0)
        r.cost_usd = self.cost(r.input_tokens, r.output_tokens)
        return r


PROVIDERS = {"jev": JevProvider, "gemini": GeminiProvider}


def load_providers(cfg_models: list[dict], force_sim: bool = False) -> dict[str, Provider]:
    out: dict[str, Provider] = {}
    for m in cfg_models:
        real = PROVIDERS[m["provider"]](m)
        out[m["id"]] = SimulatedProvider(m, real) if (force_sim or not real.has_key) else real
    return out
