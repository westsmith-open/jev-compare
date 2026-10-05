"""One live call per model on one item: prints the parsed result and the raw response.

uv run python -m jev_compare.probe [task_id] [item_id]
"""

import asyncio
import json
import sys

import httpx

from jev_compare import providers
from jev_compare.providers import load_providers
from jev_compare.server import load_models, load_tasks


async def _single_post(client, url, **kw):  # no retries: exactly one request per model
    return await client.post(url, **kw)


async def main(task_id: str = "phishing_triage", item_id: str = "ph-03") -> None:
    providers.post_with_retry = _single_post
    task = load_tasks()[task_id]
    item = next(i for i in task["items"] if i["id"] == item_id)
    async with httpx.AsyncClient(timeout=60) as client:
        for pid, prov in load_providers(load_models()).items():
            r = (await prov.classify(client, task, item)).to_dict()
            print(f"\n===== {pid} ({getattr(prov, 'model', '?')}, simulated={r['simulated']}) =====")
            print(json.dumps({k: v for k, v in r.items() if k != "raw"}, indent=2))
            print("labels:", item["labels"])
            print("raw:", json.dumps(r["raw"], indent=2)[:3000])


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
