# AGENTS.md — mistral-langfuse-showcase

## Project overview

Course for learning AI agents. I personally chose to build the course around the Mistral Python SDK and Langfuse tracing to provide a hands-on, practical learning experience.

- Showcase project: Mistral Python SDK (completions + batch) with Langfuse tracing
- Python ≥ 3.11, managed with **uv** (no pip, no poetry, no conda)
- Key dependencies: `mistralai`, `langfuse`, `pydantic` v2, `polars`, `python-dotenv`

## Directory structure

```
src/ai_devs_core/   ← library modules
src/lessons/        ← runnable lesson scripts - they start with s01e01, s01e02, etc. (season 1, episode 1, etc.)
```

- ALL library code lives in `src/ai_devs_core/`.
- DO NOT create `app/`, `lib/`, `main/`, or top-level `.py` scripts. Put new core code in `src/ai_devs_core/` and runnable scripts in `src/lessons/`.
- DO NOT move or rename existing modules without explicit instruction.
- Run lesson scripts with `uv run main.py <script_name>` from the project root (for example: `uv run main.py s02e03`).
- `uv run python -m src.lessons.<script_name>` may still work for simple module lessons, but prefer `main.py` as the canonical runner.

## Tech stack & conventions

### Python

- Target: Python 3.12+
- Type hints on every function signature. Prefer `dict`, `list`, `tuple` lowercase generics.
- Use `Any` sparingly — prefer concrete types or generics.
- Pydantic v2 only. Use `model_validator`, `field_validator`, `ConfigDict`. Never import from `pydantic.v1`.
- Favor `pathlib.Path` over `os.path`.
- Docstrings: Google style, concise. Every public function must have one.

### Dependency management

- Use `uv` exclusively:
  - Add deps: `uv add <package>`
  - Run scripts: `uv run python examples/<script>.py`
  - Sync env: `uv sync`
- NEVER use `pip install`, `poetry add`, or manual `requirements.txt`.
- To run a lesson script (for example s01e01.py), do `uv run python -m src.lessons.s01e01` from the project root

### Linting & typing

- Ruff: `uv run ruff check src/`
- Line length: 100 chars.
- Ruff rules: E, F, I, UP.

## Mistral SDK patterns

- Client is created via `create_clients()` in `client.py`. Do not instantiate `Mistral()` elsewhere.
- `MISTRAL_API_KEY` is read from env (loaded by `python-dotenv`). Never hardcode keys.
- For completions: use `client.chat.complete()` (sync), `client.chat.stream()` (streaming), `client.chat.complete_async()` (async).
- For batch: use `client.files.upload()` + `client.batch.jobs.create()` (file-based) or `client.batch.jobs.create(requests=...)` (inline).
- Prefer `mistral-small-latest` as default model unless instructed otherwise.

## JobClient for Batch Processing

The `JobClient` class in `src/ai_devs_core/job_client.py` provides a high-level interface for Mistral batch jobs:

```python
from src.ai_devs_core import JobClient, Config, get_config

# Initialize
config = get_config()
job_client = JobClient(config)

# Run batch job
def message_generator(row):
    return [{"role": "user", "content": f"Process: {row['data']}"}]

result_df = job_client.batch_job(
    df=input_dataframe,
    schema=ResponseSchema,  # Pydantic model
    task="classification",
    message_generator=message_generator,
    model="mistral-small-2603"
)
```

**Features:**
- Automatic batch job creation and monitoring
- Fallback to sequential processing if batch API fails
- Response validation against Pydantic schema
- Result merging with original DataFrame
- Comprehensive error handling and logging
- Metadata columns (`_success`, `_error`) for tracking

**Usage patterns:**
- Use for processing multiple records efficiently
- Ideal for classification, tagging, and data enrichment tasks
- Handles API rate limits and retries automatically
- Returns polars DataFrame with original + response columns

## Langfuse tracing patterns

- Langfuse auto-initialises from env vars (`LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_HOST`). No manual `Langfuse()` client needed.
- Wrap every LLM call with `@observe(as_type="generation")` — this logs input, output, model params, and token usage.
- Wrap non-LLM orchestration functions with `@observe()` to create parent spans/traces.
- Always call `langfuse_context.update_current_observation()` to record:
  - `input`, `model`, `model_parameters` before the API call
  - `usage` (with `input`/`output` token counts) and `output` after the API call
- The outermost `@observe()` function creates the root Langfuse trace; inner decorated calls nest as child spans.

## Self-check

Before finishing any task, review:

1. Did I use `uv` for all dependency and script operations?
2. Are all Pydantic models using v2 syntax with strict typing?
3. Are all LLM calls wrapped with `@observe(as_type="generation")`?
4. Are all new public functions type-hinted and documented?
5. Does `uv run ruff check src/ examples/` pass?

If any answer is no, fix it before completing the task.
