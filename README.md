# jev-compare

Send the same labelled item to a decision model ([Jev](https://docs.typesafe.ai/concepts/system-one)) and a general-purpose LLM (Gemini Flash), and compare what goes in, what comes out, how long it takes and what it costs.

> Unofficial. Not affiliated with or endorsed by TypeSafe AI or Google.

## Why

Decision models such as Jev answer typed questions (yes/no, pick one, ordered scale) and return a probability for every allowed answer in one fast pass. The usual alternative is a general LLM with a prompt and a JSON schema. This project puts the two side by side on the same items, so you can see:

- **the requests:** Jev takes the item as data plus typed questions; the LLM needs those questions written out as a prompt, plus a schema to keep its answers in bounds
- **the responses:** probabilities for every option from Jev, against a self-reported confidence from the LLM
- **time and cost** for each call, as billed
- **accuracy and calibration** across a batch, including how many items could be routed automatically at a given confidence threshold

## Quick start

You need [uv](https://docs.astral.sh/uv/) and an [OpenRouter](https://openrouter.ai) API key, which runs both models.

```sh
cp .env.example .env        # add OPENROUTER_API_KEY
uv run uvicorn jev_compare.server:app --port 8765
```

Open <http://localhost:8765>.

Without a key, both models run in **simulated** mode so you can try the interface. Simulated results are clearly labelled and their numbers are made up.

## Pages

| URL | What it does |
|---|---|
| `/` | **One call.** Pick an item, send exactly one request to each model (no retries), and see both requests and responses with syntax highlighting, plus time, cost, tokens and correct answers. |
| `/bench` | **Batch.** Run a whole task set through both models: accuracy per question, latency spread, cost per item and per million items, and a confidence-threshold chart. Runs are saved to `runs/`. |

To check your setup from the command line with one call per model:

```sh
uv run python -m jev_compare.probe phishing_triage ph-03
```

## How each model is called

| | Jev | Gemini |
|---|---|---|
| Endpoint | OpenRouter Decisions API, `POST /api/alpha/decisions` | OpenRouter chat completions, `POST /api/v1/chat/completions` |
| Input | `state` (the item) and typed `questions` | A markdown prompt holding the item and the questions, plus a strict JSON schema |
| Output | An answer per question with probabilities for every option | JSON with an answer and a self-reported confidence per question |
| Settings | None | `temperature: 0`, `reasoning: minimal` (the closest an LLM gets to a single fast pass) |

Both models get exactly the same information. Only the packaging differs.

Both adapters are in [`jev_compare/providers.py`](jev_compare/providers.py). Set `TYPESAFE_API_KEY` or `GEMINI_API_KEY` to call the vendors directly instead of through OpenRouter.

## Task sets

Each file in `tasks/` defines one task: a list of questions and a set of items with known answers. All items are synthetic.

| Task | Items | Questions |
|---|---|---|
| Phishing report triage | 24 | `is_phishing` (yes/no), `lure` (choice), `risk` (scale) |
| Job posting tagging | 24 | `work_location` (choice), `seniority` (scale), `function` (choice), `visa_sponsorship` (yes/no) |
| Recipe dietary tagging | 24 | `diet` (choice), `contains_gluten` (yes/no), `contains_nuts` (yes/no), `spice` (scale) |

The sets are small and deliberately include hard cases: lookalike domains, genuine emails that look urgent, quarterly office visits on "remote" jobs, and soy sauce versus tamari. With 24 items, one answer changes accuracy by about 4 points, so treat the results as a demonstration rather than a benchmark.

### Writing your own task

```json
{
  "id": "my_task",
  "title": "My task",
  "annual_volume": 100000,
  "description": "What the items are and what the questions decide.",
  "questions": [
    {"id": "q1", "type": "noul", "instructions": "A statement that is true or false.",
     "criteria": {"true": "When it is true", "false": "When it is false"}},
    {"id": "q2", "type": "choice", "instructions": "Which option?",
     "criteria": {"a": "Description of a", "b": "Description of b"}},
    {"id": "q3", "type": "score", "instructions": "How much?",
     "levels": [{"key": "low", "text": "Low: ..."}, {"key": "high", "text": "High: ..."}]}
  ],
  "items": [
    {"id": "x-01", "state": {"text": "..."}, "labels": {"q1": true, "q2": "a", "q3": "low"}}
  ]
}
```

The question types follow Jev's primitives: [noul](https://docs.typesafe.ai/primitives/noul), [choice](https://docs.typesafe.ai/primitives/choice) and [score](https://docs.typesafe.ai/primitives/score). Any context that every item needs, such as a policy, can go in `shared_state`; it is merged into each item's `state`.

To keep private task sets outside this repository, point the server at another folder:

```sh
JEV_COMPARE_TASKS=/path/to/my/tasks JEV_COMPARE_RUNS=/path/to/my/runs uv run uvicorn jev_compare.server:app
```

## Models and prices

Models are listed in [`config/models.json`](config/models.json). OpenRouter returns the billed cost with each response, and that figure is used when present. The prices in the config are only a fallback. Gemini 3.8 Flash is at its introductory price until 31 December 2026.

## Contributing

Linting and formatting use [ruff](https://docs.astral.sh/ruff/), run through [pre-commit](https://pre-commit.com):

```sh
uv sync                          # installs ruff and pre-commit
uv run pre-commit install        # run the hooks on every commit
uv run pre-commit run --all-files
```

The hooks also check JSON, TOML and YAML syntax, fix trailing whitespace and line endings, and block commits that contain a private key.

## Licence

MIT
