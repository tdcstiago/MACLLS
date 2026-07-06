# 🔍 MACLLS — Multi-Agent Contrastive Language-Learning System

MACLLS analyzes a **word or a full sentence** in a learner's native language (L1) and
produces a rich, CEFR-calibrated contrastive lesson in a target language (L2) — surfacing
**false friends**, correct translations, structural transfer errors, and idiomatic usage.
A team of specialized LLM agents does the linguistic reasoning; a local MCP server supplies
authoritative curated data and spaCy-based structural parsing; and everything is persisted in
a local SQLite database that also powers a built-in **spaced-repetition flashcard system**.

---

## ✨ Features

- **Hybrid input** — type a single word (lexical analysis) or a whole sentence (syntactic /
  structural analysis); the system auto-detects which.
- **Multi-agent pipeline** — an L1 Specialist and L2 Specialist run in parallel, then a
  Pedagogue Bridge agent synthesizes a single structured lesson.
- **False-friend detection** — a curated knowledge base flags dangerous look-alikes
  (e.g. PT *pretender* → EN *pretend* ✗ / *intend* ✓), and scans for them inside sentences.
- **CEFR calibration (A1–C2)** — vocabulary, tone, grammatical depth, and number of worked
  examples scale to the selected proficiency level.
- **spaCy structural parsing** — POS tags, dependency relations, and word-order transfer cues
  ground the sentence analysis (degrades gracefully to LLM-only when a model isn't installed).
- **Strict structured output** — the Pedagogue returns a Pydantic `response_schema`-validated
  JSON payload, so the UI never has to parse malformed text.
- **Automatic response cache** — identical requests return instantly from SQLite (30-day TTL);
  prompt changes bust the cache automatically via a version tag.
- **Spaced-repetition flashcards (SM-2)** — save any analysis as a flashcard and review it in a
  session-based "Daily Practice" tab with a progress tracker.
- **Resilient by design** — exponential-backoff retries, a fallback model, empty-response
  guards, prompt-injection delimiters, and a mock mode that works with no API key.

---

## 🧠 Architecture

```
                        ┌────────────────────────────┐
   User (Streamlit UI)  │  app.py                     │
        │               │  Tab 1: Contrastive Analysis│
        │  word/sentence │  Tab 2: Daily Practice (SRS)│
        ▼               └─────────────┬──────────────┘
┌───────────────────────┐            │ process_lesson()
│ LanguageOrchestrator   │◄───────────┘
│ agents/orchestrator.py │
│                        │   1. classify: word | sentence
│                        │   2. MCP lookup (local)  ──────────┐
│                        │   3. L1 + L2 specialists (parallel)│
│                        │   4. Pedagogue synthesis (JSON)    │
└──────┬─────────┬───────┘                                    │
       │         │                                            ▼
       │         │ Gemini API                    ┌────────────────────────────┐
       │         ▼ (google-genai)                │ mcp_servers/                │
       │  gemini-2.5-flash (primary)             │ linguistics_server.py       │
       │  gemini-2.5-pro   (fallback)            │  • discover_contrastive_    │
       │                                         │    scenarios (word)         │
       ▼                                         │  • analyze_sentence_        │
┌───────────────────────┐                        │    structure (spaCy)        │
│ database/db_manager.py │                        │  • curated false-friend DB  │
│  SQLite: maclls_local.db│                       └────────────────────────────┘
│  • lesson_cache (TTL)  │
│  • managed_languages   │        ┌────────────────────────┐
│  • system_settings     │        │ database/srs_engine.py │
│  • flashcards ─────────┼───────►│ SM-2 scheduling math   │
└───────────────────────┘        └────────────────────────┘
```

### The three agents (`agents/prompts.py`)

| Agent | Role | Word mode | Sentence mode |
|-------|------|-----------|---------------|
| **L1 Specialist** | Diagnoses native-language interference (internal, not shown to user) | Lexical false-friend / cognate confusion | Syntactic transfer: word order, prepositions, tense/aspect, calques |
| **L2 Specialist** | Provides the flawless native-speaker model (CEFR-register aware) | Correct word + idiomatic usage | Idiomatic full-sentence rendering + alternatives |
| **Pedagogue Bridge** | Synthesizes the final user-facing lesson (structured JSON, written in L1) | Safe/Dangerous sections + full lesson body | Structural-contrast lesson |

The L1 and L2 specialists execute **concurrently** (`ThreadPoolExecutor`, fail-fast); the
Pedagogue runs once and returns a schema-validated payload.

### 🔌 MCP server

`mcp_servers/linguistics_server.py` is a fully standards-compliant **MCP server** built on
`FastMCP`, exposing two tools (`discover_contrastive_scenarios`, `analyze_sentence_structure`).

> **Note on execution:** the Streamlit app imports these tools **directly** (in-process) for
> lowest latency. That is a deliberate performance choice — it is *not* a limitation of the MCP
> layer. The same server can be run and consumed over **stdio** by any MCP client:
>
> ```bash
> mcp run mcp_servers/linguistics_server.py
> ```

---

## 📁 Project structure

```
MACLLS/                           # repository root
├── app.py                        # Streamlit UI: sidebar, Analysis tab, Daily Practice tab
├── agents/
│   ├── orchestrator.py           # LanguageOrchestrator: pipeline, retries, cache, schemas
│   └── prompts.py                # Agent personas, CEFR profiles, focus blocks, lesson skeletons
├── mcp_servers/
│   ├── linguistics_server.py     # FastMCP tools: curated false friends + spaCy sentence parse
│   └── spacy_manager.py          # Thread-safe Singleton: load spaCy models once, NER excluded
├── observability/
│   ├── logger_setup.py           # JSON-to-stdout logging (LOG_LEVEL-driven, idempotent)
│   └── correlation.py            # Per-request correlation ID (ContextVar + logging.Filter)
├── database/
│   ├── db_manager.py             # SQLite: settings, languages, lesson cache, flashcards
│   └── srs_engine.py             # Pure SM-2 spaced-repetition math
├── scripts/
│   └── smoke_test.py             # Quick "is the Gemini key working?" check (reads env var)
├── tests/                        # 88 tests (unittest)
│   ├── test_agents_prompts.py
│   ├── test_orchestrator.py
│   ├── test_orchestrator_api.py  # retry/fallback/parallel/schema with a fake client
│   ├── test_linguistics_server.py
│   ├── test_spacy_manager.py     # Singleton identity, thread-safe load-once, NER-excluded
│   ├── test_logging.py           # JSON schema, correlation ID, copy_context thread propagation
│   ├── test_database.py
│   └── test_srs.py
├── .streamlit/
│   ├── secrets.toml              # LOCAL ONLY, git-ignored — your real GEMINI_API_KEY
│   └── secrets.toml.example      # template
├── setup.sh                      # cloud post-build: pip install + spaCy models
├── requirements.txt
├── pyproject.toml                # pytest config (testpaths = ["tests"])
├── README.md
└── maclls_local.db               # created at runtime, git-ignored
```

---

## 🛠️ Tech stack

- **Python** (developed against 3.12+/3.14) — stdlib `sqlite3`, `concurrent.futures`, `dataclasses`
- **[Streamlit](https://streamlit.io/)** — presentation layer
- **[google-genai](https://pypi.org/project/google-genai/) `~=1.0`** — modern Gemini SDK
  (models: `gemini-2.5-flash` primary, `gemini-2.5-pro` fallback)
- **[mcp](https://pypi.org/project/mcp/) `~=1.0`** — FastMCP local linguistics server
- **[spaCy](https://spacy.io/) `~=3.7`** — POS tagging & dependency parsing (optional models)
- **[Pydantic](https://docs.pydantic.dev/)** — strict `response_schema` for structured LLM output

---

## 🚀 Setup

> Commands below are cross-platform. Activate the virtual environment once, then run
> everything through it. **Windows** users activate with `venv\Scripts\activate` instead of
> `source venv/bin/activate`.

### 1. Create a virtual environment & install dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Install spaCy models for sentence parsing

Language models are **not** pip dependencies. Sentence analysis works without them (LLM-only
fallback), but for real POS/dependency parsing install the ones you need:

```bash
python -m spacy download pt_core_news_sm en_core_web_sm
# es_core_news_sm · fr_core_news_sm · de_core_news_sm · it_core_news_sm · ro_core_news_sm
```

For a cloud deployment, `setup.sh` installs the dependencies and all 7 models in one step.

### ☁️ Deployability — Docker-ready

The project is **Docker-ready** and ships a full production stack: `Dockerfile`,
`docker-compose.yml`, `Procfile`, `.dockerignore`, and a `deploy.sh` helper. The
`Dockerfile` (`python:3.12-slim`) installs dependencies and bakes in the spaCy models at build
time, exposes `8501`, and launches Streamlit; `.dockerignore` keeps secrets, the venv, and local
DBs out of the image.

**Method A — Docker Compose (recommended):**

```bash
export GEMINI_API_KEY="your-key"          # or put it in a local .env file
docker-compose up -d --build              # builds the image and starts the app on :8501
```

`deploy.sh` wraps this for a server (`git pull` → rebuild → reload Nginx):

```bash
./deploy.sh
```

> **Note:** The `deploy.sh` script is designed for production Linux environments utilizing Nginx
> and systemd. It is not intended for serverless platforms like Render or Heroku.

**Method B — Nginx reverse proxy** (put the container behind a domain / TLS). Point Nginx at the
container's exposed port:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;      # Streamlit needs WebSocket upgrade
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

```bash
sudo nginx -t && sudo systemctl reload nginx
```

**Buildpack platforms** (Render, Heroku) use the `Procfile`:
`web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

> **MCP honesty note:** `mcp_servers/linguistics_server.py` is a fully standards-compliant MCP
> server (`FastMCP`, `mcp.run()`). The web app **imports its tools directly (in-process)** for
> lowest latency — a deliberate performance choice, not a limitation. The same server can be run
> and consumed over stdio by any MCP client: `mcp run mcp_servers/linguistics_server.py`.

### 3. Configure your Gemini API key

Copy the template and paste your key (get one from Google AI Studio):

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # Windows: use `copy` and backslashes
# then edit .streamlit/secrets.toml → GEMINI_API_KEY = "your-key"
```

`secrets.toml` and `*.db` are git-ignored — **never commit real keys**. You can also paste the
key directly into the sidebar at runtime. Without any key the app still runs in **mock mode**
(local false-friend data only, no LLM lesson).

---

## ▶️ Running the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`).

### Sidebar
- **Native (L1)** and **Target (L2)** languages — populated from the DB language registry.
- **CEFR Proficiency Level** — A1 → C2.
- **Gemini API Key** — auto-loaded from `secrets.toml` if present.
- **🧹 Limpar cache de lições** — clears all cached lessons.
- Your selections persist across restarts (saved in `system_settings`).

### Tab 1 — 🔍 Contrastive Analysis
Type a word or sentence and click **Analisar**. You get a CEFR badge, Safe/Dangerous cards (word
mode) or an idiomatic rendering + structural notes (sentence mode), the full pedagogical lesson,
and related suggestions. Click **💾 Save to Daily Practice** to turn it into a flashcard.

### Tab 2 — 🧠 Daily Practice
Click **▶️ Start Review Session** to pull all due cards (shuffled). A progress bar shows
*Cards Reviewed / Remaining*. Reveal the answer, then grade **Again · Hard · Good · Easy** — the
SM-2 engine reschedules the card, "Again" re-queues it for later in the session, and an empty
queue shows **🎉 All caught up for today!**

### Terminal CLI

The same multi-agent pipeline is also exposed as an **Agents CLI** for headless / scripted use.
It reads `GEMINI_API_KEY` from the environment or `.streamlit/secrets.toml`, reuses the SQLite
cache, and falls back to mock mode with no key.

```bash
python cli.py "pretender"                          # word (defaults: PT→EN, B1)
python cli.py "Eu pretendo viajar" --cefr C1       # sentence, advanced level
python cli.py "casa" --l1 Portuguese --l2 English  # explicit languages
```

Flags: positional `text`, `--l1` (default Portuguese), `--l2` (default English), `--cefr`
(default B1). It prints the CEFR level, the Safe/Dangerous targets (word mode), and the full
lesson markdown.

---

## 💾 Data & persistence (`maclls_local.db`)

| Table | Purpose |
|-------|---------|
| `system_settings` | Persisted UI preferences (L1, L2, CEFR level) |
| `managed_languages` | Language registry (code, display name, spaCy model, active) — seeded with 7 languages on first run |
| `lesson_cache` | Cached Pedagogue responses, keyed by `sha256(l1\|l2\|level\|mode\|text\|PROMPT_VERSION)`, with a **30-day TTL** |
| `flashcards` | SRS cards: `front_text`, `back_text`, `next_review`, `interval`, `ease_factor`, `repetitions` |

**Caching:** the orchestrator checks `lesson_cache` before any LLM/MCP work; a fresh hit is
returned instantly. Because `PROMPT_VERSION` is part of the key, changing prompt logic
transparently invalidates stale entries — no manual clear needed.

**Spaced repetition (SM-2, `srs_engine.py`):** the four UI grades map to SM-2 quality values
(Again→0, Hard→3, Good→4, Easy→5). Intervals grow 1 → 6 → `round(interval × ease_factor)`;
the ease factor floors at 1.3; a lapse (Again) resets repetitions and interval.

**Supported languages:** English, Portuguese, Spanish, French, German, Italian, Romanian.

---

## 🧪 Testing

```bash
python -m unittest discover -s tests
```

88 tests cover: prompt integrity, the orchestrator's retry/fallback/parallel/empty-response
logic (via an injected fake Gemini client — no network needed), input classification, the MCP
false-friend scan and graceful spaCy fallback, the spaCy Singleton (thread-safe load-once, NER
exclusion, warmup fail-safe), the observability layer (JSON schema, correlation-ID propagation
through `copy_context`), the SQLite layer (settings/cache/TTL/languages/token accounting), and
the SM-2 math + flashcard scheduling. Tests use in-memory SQLite and never require an API key or
spaCy models (model-dependent assertions skip cleanly when a model isn't installed).

---

## 🔐 Security & resilience notes

- **Secrets** are git-ignored (`.streamlit/secrets.toml`, `*.db`); the app treats a leftover
  placeholder key as "missing."
- **Prompt injection** — user words/sentences are delimited (`<<< >>>`) and length-capped
  (100 chars for words, 500 for sentences); prompts instruct the model to treat them as data.
- **Retries** — transient HTTP errors (429/500/502/503/504) get exponential backoff, then a
  fallback to `gemini-2.5-pro`; empty/blocked responses raise a clear error instead of failing
  silently.
- **Thread safety & concurrency** — the SQLite connection uses `check_same_thread=False` with a
  write lock, and runs in **WAL** (Write-Ahead Logging) mode to avoid "database is locked"
  errors under concurrent web requests.

---

## 💰 Cost & observability

- **Token usage tracking** — before each generation the `LanguageOrchestrator` counts the prompt
  tokens (`count_tokens`) and persists them as `token_count` on the `lesson_cache` row, so LLM
  spend is observable directly from SQLite. It's best-effort — token accounting never blocks a
  lesson.
- **Graceful parse auditing** — every LLM payload carries a `parse_ok` flag. When the JSON can't
  be parsed (malformed output or a prompt-injection escape attempt), the app shows a friendly
  "try again" warning instead of crashing, the orchestrator refuses to cache the bad response,
  and the first 100 characters of the offending payload are logged via `logger.warning` for
  security auditing.

---

## 📡 Observability & Telemetry

MACLLS emits **structured JSON logs to `stdout`**, so a container runtime (Docker, OCI, or an
Nginx-fronted host) captures the full request trace with `docker logs` / `oci logging` — no log
files, no extra agent. The observability layer lives in `observability/` and is wired only at the
two entry points (`app.py`, `cli.py`); the domain layers (orchestrator, MCP tools) stay decoupled
and just call the stdlib `logging` module — Clean Architecture is preserved.

### Log schema

Every line is a self-contained JSON object:

```json
{
  "timestamp": "2026-07-03T14:22:05.512Z",
  "level": "INFO",
  "logger": "agents.orchestrator",
  "message": "agent.done",
  "correlation_id": "ff0bd9e9c1d4487aa1b2c3d4e5f60718",
  "agent": "L1",
  "duration_ms": 14537,
  "token_count": 1791
}
```

| Field | Always present | Meaning |
|-------|:---:|---------|
| `timestamp` | ✓ | UTC ISO-8601, millisecond precision |
| `level` | ✓ | `INFO` / `WARNING` / `ERROR` / `DEBUG` |
| `logger` | ✓ | Emitting module (e.g. `agents.orchestrator`, `mcp_servers.linguistics_server`) |
| `message` | ✓ | Event name (see below) |
| `correlation_id` | ✓ | Per-request trace ID (`-` when outside a request) |
| `error` | on exception | Formatted exception + traceback |
| *extras* | per-event | `agent`, `duration_ms`, `token_count`, `input_mode`, `l1`, `l2`, `cefr`, `parse_ok`, `input` (truncated), … |

**Event vocabulary** (the `message` field): `request.start` → `cache.hit` / `cache.miss` →
`mcp.discover` / `mcp.analyze_sentence` → `agent.start` / `agent.done` (`L1`, `L2`, `pedagogue`) →
`llm.retry` / `llm.fallback` → `cache.store` → `request.done`.

### Correlation ID — tracing one request across the multi-agent pipeline

A single request fans out across parallel agents (L1 + L2 in a `ThreadPoolExecutor`), the spaCy
MCP tools, multiple Gemini calls, and the cache. To stitch those scattered log lines back into
**one traceable story**, each entry point generates a `correlation_id` (uuid4 hex) and stamps it
on every log record via a `contextvars.ContextVar` + a `logging.Filter` — no function signature
carries it, so the layer boundaries stay clean.

The one subtlety: `ContextVar`s do **not** auto-propagate into `ThreadPoolExecutor` worker
threads, so the L1/L2 specialists would otherwise lose the ID. We fix that by copying the context
into each worker (`contextvars.copy_context().run(...)`). The payoff — every line of a request,
including the parallel specialists, shares one ID:

```bash
$ LOG_LEVEL=INFO python cli.py "livro" | grep correlation_id
{"message":"request.start","correlation_id":"ff0bd9e9…","input_mode":"word","input":"livro"}
{"message":"agent.start","correlation_id":"ff0bd9e9…","agent":"L1"}   # worker thread
{"message":"agent.start","correlation_id":"ff0bd9e9…","agent":"L2"}   # worker thread
{"message":"agent.done","correlation_id":"ff0bd9e9…","agent":"pedagogue","duration_ms":7511,"token_count":1791}
{"message":"request.done","correlation_id":"ff0bd9e9…","duration_ms":22152,"parse_ok":true}
```

### `LOG_LEVEL` — controlling verbosity (Docker / OCI)

Verbosity is driven by the `LOG_LEVEL` environment variable (`DEBUG` / `INFO` / `WARNING` /
`ERROR`), so you tune it **per environment without a code change** — exactly what you want in a
container:

```bash
docker run -e LOG_LEVEL=INFO    maclls   # full request trace (default for the web app)
docker run -e LOG_LEVEL=DEBUG   maclls   # + verbose diagnostics
docker run -e LOG_LEVEL=WARNING maclls   # only retries / fallbacks / errors
```

```yaml
# docker-compose.yml
services:
  maclls:
    environment:
      - LOG_LEVEL=INFO
```

Defaults: the **Streamlit app** logs at `INFO` (server stdout, no UI clutter); the **CLI** defaults
to `WARNING` so `python cli.py "peixe"` prints a clean lesson — set `LOG_LEVEL=INFO` to see the
full JSON trace interleaved. Noisy third-party loggers (`httpx`, `google_genai`, `spacy`, …) are
pinned to `WARNING` so the stream stays readable.

**Security guardrail:** logs never contain the API key, full user input (truncated to 40 chars),
or lesson bodies — only metadata, timings, and token counts safe to ship to a log aggregator.

---

## ⚡ Performance — spaCy model Singleton

spaCy pipelines are heavy to load, so `mcp_servers/spacy_manager.py` manages them as a
**process-wide Singleton**: each language's model is loaded **once** and shared *by reference*
across every request, never reloaded per call.

- **Load once, share by reference** — a language's `nlp` object is built on first use and cached
  for the life of the container/process; subsequent requests reuse the same in-memory instance.
- **Thread-safe (double-checked locking)** — the hot path is an unlocked dict read; only a cache
  miss takes a lock, and re-checks inside it, so the parallel L1/L2 specialists can never
  double-load the same model.
- **NER excluded** — models load with `exclude=["ner"]`. We only consume POS (tagger/
  morphologizer), dependency parse (parser), and lemmas (lemmatizer); dropping the unused NER
  component lowers RAM and speeds up both load and per-document inference.
- **Eager warmup** — `app.py` and `cli.py` call `warmup(l1)` at startup to preload the native
  language, so a demo's first sentence parse isn't slow. Warmup is **fail-safe**: if the model
  isn't installed it logs a single `spacy.warmup_skipped` warning and the app starts normally
  (mock / model-less deployments are unaffected).

A cold load emits one structured `spacy.model_loaded` log line (carrying the request's correlation
ID) — you can confirm in the logs that a model loads exactly once per process and never again.

---

## 📌 Notes & limitations

- The curated false-friend DB is Portuguese→English focused; other language pairs rely on the
  LLM (with spaCy for structure). Extend `CONTRASTIVE_DB` in `linguistics_server.py` to add pairs.
- Pedagogical section headers are in Portuguese by default; the Safe/Dangerous card labels are
  localized to the L1 across all 7 languages.
- The lesson cache is per-input/level/prompt-version; bump `PROMPT_VERSION` in `orchestrator.py`
  after any prompt change you want reflected immediately.
