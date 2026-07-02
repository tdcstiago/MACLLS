"""Unified local persistence for MACLLS (Python stdlib sqlite3 only).

Handles three concerns behind one connection:
  * lesson_cache      — automated LLM response caching with a TTL
  * managed_languages — dynamic language registry (seeded on first run)
  * system_settings   — persisted user preferences (L1/L2/CEFR)
"""

import hashlib
import json
import sqlite3
import threading
from datetime import date, datetime, timedelta

from database.srs_engine import review as srs_review, DEFAULT_EASE_FACTOR

DEFAULT_DB_PATH = "maclls_local.db"

# The 7 core languages seeded on first run: (code, display_name, spacy_model, is_active).
# Mirrors SPACY_MODELS in mcp_servers/linguistics_server.py (kept in sync deliberately).
CORE_LANGUAGES = [
    ("en", "English", "en_core_web_sm", 1),
    ("pt", "Portuguese", "pt_core_news_sm", 1),
    ("es", "Spanish", "es_core_news_sm", 1),
    ("fr", "French", "fr_core_news_sm", 1),
    ("de", "German", "de_core_news_sm", 1),
    ("it", "Italian", "it_core_news_sm", 1),
    ("ro", "Romanian", "ro_core_news_sm", 1),
]


class DatabaseManager:
    """A thin, thread-safe wrapper around a single sqlite3 connection."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        # check_same_thread=False: Streamlit reruns may touch the connection from
        # different threads; a Lock guards all writes.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        # Write-Ahead Logging: lets readers and a writer work concurrently, avoiding
        # "database is locked" errors under concurrent web requests. (No-op on :memory:.)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._create_tables()
        self._migrate()
        self._seed_languages()

    # --- schema + seeding ----------------------------------------------------

    def _create_tables(self) -> None:
        with self._lock:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS system_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
                CREATE TABLE IF NOT EXISTS managed_languages (
                    code TEXT PRIMARY KEY,
                    display_name TEXT,
                    spacy_model TEXT,
                    is_active INTEGER
                );
                CREATE TABLE IF NOT EXISTS lesson_cache (
                    cache_key TEXT PRIMARY KEY,
                    input_mode TEXT,
                    proficiency_level TEXT,
                    response_json TEXT,
                    created_at DATETIME,
                    token_count INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS flashcards (
                    id INTEGER PRIMARY KEY,
                    cache_key TEXT,
                    front_text TEXT,
                    back_text TEXT,
                    next_review DATE,
                    interval INTEGER,
                    ease_factor REAL,
                    repetitions INTEGER
                );
                """
            )
            self.conn.commit()

    def _migrate(self) -> None:
        """Additive schema migrations for pre-existing databases (safe to re-run)."""
        with self._lock:
            try:
                self.conn.execute(
                    "ALTER TABLE lesson_cache ADD COLUMN token_count INTEGER DEFAULT 0;"
                )
                self.conn.commit()
            except sqlite3.OperationalError:
                pass  # column already exists (fresh DB or already migrated)

    def _seed_languages(self) -> None:
        """Idempotently seed the core languages (existing rows are left untouched)."""
        with self._lock:
            self.conn.executemany(
                "INSERT OR IGNORE INTO managed_languages "
                "(code, display_name, spacy_model, is_active) VALUES (?, ?, ?, ?)",
                CORE_LANGUAGES,
            )
            self.conn.commit()

    # --- settings ------------------------------------------------------------

    def get_setting(self, key: str, default=None):
        row = self.conn.execute(
            "SELECT value FROM system_settings WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row is not None else default

    def set_setting(self, key: str, value) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO system_settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )
            self.conn.commit()

    # --- languages -----------------------------------------------------------

    def get_active_languages(self) -> list:
        rows = self.conn.execute(
            "SELECT code, display_name, spacy_model FROM managed_languages "
            "WHERE is_active = 1 ORDER BY display_name"
        ).fetchall()
        return [dict(row) for row in rows]

    # --- cache ---------------------------------------------------------------

    def store_cache(
        self,
        cache_key: str,
        input_mode: str,
        proficiency_level: str,
        response_json: str,
        token_count: int = 0,
        created_at=None,
    ) -> None:
        """Upsert a cache entry (with prompt token count for cost tracking).
        `created_at` accepts a datetime or ISO string; the optional parameter
        exists mainly so tests can inject old timestamps."""
        ts = created_at if created_at is not None else datetime.now()
        if isinstance(ts, datetime):
            ts = ts.isoformat()
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO lesson_cache "
                "(cache_key, input_mode, proficiency_level, response_json, created_at, token_count) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (cache_key, input_mode, proficiency_level, response_json, ts, token_count),
            )
            self.conn.commit()

    def get_cached(self, cache_key: str, ttl_days: int = 30):
        """Return the parsed cached object (with its `token_count` attached), or
        None if missing or older than TTL."""
        row = self.conn.execute(
            "SELECT response_json, created_at, token_count FROM lesson_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        try:
            created = datetime.fromisoformat(row["created_at"])
        except (TypeError, ValueError):
            return None
        if datetime.now() - created > timedelta(days=ttl_days):
            return None
        try:
            result = json.loads(row["response_json"])
        except (TypeError, ValueError):
            return None
        if isinstance(result, dict):
            result["token_count"] = row["token_count"]
        return result

    def clear_cache(self) -> int:
        """Delete every cached lesson. Returns the number of rows removed."""
        with self._lock:
            cur = self.conn.execute("DELETE FROM lesson_cache")
            self.conn.commit()
            return cur.rowcount

    # --- flashcards (spaced repetition) --------------------------------------

    @staticmethod
    def _as_iso(day) -> str:
        return day.isoformat() if isinstance(day, date) else str(day)

    def add_flashcard(self, cache_key: str, front: str, back: str, next_review=None) -> int:
        """Create a new flashcard due immediately (unless next_review is given).
        Returns the new card id."""
        nr = self._as_iso(next_review if next_review is not None else date.today())
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO flashcards "
                "(cache_key, front_text, back_text, next_review, interval, ease_factor, repetitions) "
                "VALUES (?, ?, ?, ?, 0, ?, 0)",
                (cache_key, front, back, nr, DEFAULT_EASE_FACTOR),
            )
            self.conn.commit()
            return cur.lastrowid

    def get_due_flashcards(self, today=None) -> list:
        """Return cards whose next_review is on or before `today` (default: today)."""
        cutoff = self._as_iso(today if today is not None else date.today())
        rows = self.conn.execute(
            "SELECT * FROM flashcards WHERE next_review <= ? ORDER BY next_review, id",
            (cutoff,),
        ).fetchall()
        return [dict(row) for row in rows]

    def update_flashcard_progress(self, card_id: int, grade: int, today=None):
        """Apply an SM-2 review to a card and persist the new schedule.

        Returns the resulting SrsState, or None if the card does not exist."""
        row = self.conn.execute(
            "SELECT interval, ease_factor, repetitions FROM flashcards WHERE id = ?",
            (card_id,),
        ).fetchone()
        if row is None:
            return None

        state = srs_review(
            grade, row["repetitions"], row["ease_factor"], row["interval"], today=today
        )
        with self._lock:
            self.conn.execute(
                "UPDATE flashcards SET interval = ?, ease_factor = ?, repetitions = ?, "
                "next_review = ? WHERE id = ?",
                (state.interval, state.ease_factor, state.repetitions,
                 state.next_review.isoformat(), card_id),
            )
            self.conn.commit()
        return state

    # --- helpers -------------------------------------------------------------

    @staticmethod
    def build_cache_key(
        l1: str, l2: str, level: str, mode: str, text: str, version: str = "v1"
    ) -> str:
        """Deterministic, case-insensitive sha256 key over the request identity.

        `version` lets the caller bust the cache when prompt logic changes (a
        different version yields a different key for the same request)."""
        raw = "|".join(
            str(part).lower() for part in (l1, l2, level, mode, text, version)
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def close(self) -> None:
        self.conn.close()
