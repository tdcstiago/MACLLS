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
│   └── linguistics_server.py     # FastMCP tools: curated false friends + spaCy sentence parse
├── database/
│   ├── db_manager.py             # SQLite: settings, languages, lesson cache, flashcards
│   └── srs_engine.py             # Pure SM-2 spaced-repetition math
├── scripts/
│   └── smoke_test.py             # Quick "is the Gemini key working?" check (reads env var)
├── tests/                        # 65 tests (unittest)
│   ├── test_agents_prompts.py
│   ├── test_orchestrator.py
│   ├── test_orchestrator_api.py  # retry/fallback/parallel/schema with a fake client
│   ├── test_linguistics_server.py
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

### ☁️ One-click cloud deployment

The project ships with both a **`Dockerfile`** and a **`Procfile`** for true one-click deploys:

- **`Dockerfile`** (`python:3.12-slim`) installs dependencies, runs `setup.sh` to bake in the
  spaCy models at build time, exposes `8501`, and launches Streamlit — deployable to any
  container host (AWS, Cloud Run, Fly.io, …).
- **`Procfile`** (`web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0`)
  targets buildpack platforms like Render and Heroku.

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

65 tests cover: prompt integrity, the orchestrator's retry/fallback/parallel/empty-response
logic (via an injected fake Gemini client — no network needed), input classification, the MCP
false-friend scan and graceful spaCy fallback, the SQLite layer (settings/cache/TTL/languages/
token accounting), and the SM-2 math + flashcard scheduling. Tests use in-memory SQLite and
never require an API key or spaCy models.

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

## 📌 Notes & limitations

- The curated false-friend DB is Portuguese→English focused; other language pairs rely on the
  LLM (with spaCy for structure). Extend `CONTRASTIVE_DB` in `linguistics_server.py` to add pairs.
- Pedagogical section headers are in Portuguese by default; the Safe/Dangerous card labels are
  localized to the L1 across all 7 languages.
- The lesson cache is per-input/level/prompt-version; bump `PROMPT_VERSION` in `orchestrator.py`
  after any prompt change you want reflected immediately.
