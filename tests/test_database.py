import json
import unittest
from datetime import datetime, timedelta

from database.db_manager import DatabaseManager


class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        # In-memory DB: no files, isolated per test.
        self.db = DatabaseManager(":memory:")

    def tearDown(self):
        self.db.close()

    # --- language seeding ----------------------------------------------------

    def test_seeds_seven_active_languages(self):
        langs = self.db.get_active_languages()
        self.assertEqual(len(langs), 7)
        by_name = {row["display_name"]: row for row in langs}
        self.assertEqual(by_name["English"]["spacy_model"], "en_core_web_sm")
        self.assertEqual(by_name["Portuguese"]["spacy_model"], "pt_core_news_sm")
        self.assertIn("Romanian", by_name)

    def test_seeding_is_idempotent(self):
        self.db._seed_languages()
        self.db._seed_languages()
        self.assertEqual(len(self.db.get_active_languages()), 7)

    # --- settings ------------------------------------------------------------

    def test_setting_returns_default_when_absent(self):
        self.assertEqual(self.db.get_setting("l1_lang", "Portuguese"), "Portuguese")
        self.assertIsNone(self.db.get_setting("missing"))

    def test_setting_upsert_overwrites(self):
        self.db.set_setting("cefr_level", "A1")
        self.assertEqual(self.db.get_setting("cefr_level"), "A1")
        self.db.set_setting("cefr_level", "C2")  # ON CONFLICT update
        self.assertEqual(self.db.get_setting("cefr_level"), "C2")

    # --- cache key -----------------------------------------------------------

    def test_build_cache_key_deterministic_and_case_insensitive(self):
        k1 = DatabaseManager.build_cache_key("Portuguese", "English", "B1", "word", "Pretender")
        k2 = DatabaseManager.build_cache_key("portuguese", "english", "b1", "word", "pretender")
        self.assertEqual(k1, k2)
        self.assertEqual(len(k1), 64)  # sha256 hex

    def test_build_cache_key_distinguishes_inputs(self):
        base = DatabaseManager.build_cache_key("Portuguese", "English", "B1", "word", "pretender")
        self.assertNotEqual(base, DatabaseManager.build_cache_key("Portuguese", "English", "B1", "word", "peixe"))
        self.assertNotEqual(base, DatabaseManager.build_cache_key("Portuguese", "English", "C2", "word", "pretender"))

    def test_build_cache_key_version_busts_cache(self):
        args = ("Portuguese", "English", "B1", "word", "pretender")
        self.assertNotEqual(
            DatabaseManager.build_cache_key(*args, version="1"),
            DatabaseManager.build_cache_key(*args, version="2"),
        )
        # Same version → still deterministic.
        self.assertEqual(
            DatabaseManager.build_cache_key(*args, version="2"),
            DatabaseManager.build_cache_key(*args, version="2"),
        )

    # --- cache store / retrieve ---------------------------------------------

    def test_cache_store_and_retrieve_roundtrip(self):
        payload = json.dumps({"lesson": "oi", "alternatives": ["a", "b"], "has_local_data": True})
        self.db.store_cache("k1", "word", "B1", payload)
        got = self.db.get_cached("k1")
        self.assertEqual(got["lesson"], "oi")
        self.assertEqual(got["alternatives"], ["a", "b"])
        self.assertTrue(got["has_local_data"])

    def test_cache_miss_returns_none(self):
        self.assertIsNone(self.db.get_cached("does-not-exist"))

    def test_lesson_cache_has_token_count_column(self):
        cols = {r[1] for r in self.db.conn.execute("PRAGMA table_info(lesson_cache)").fetchall()}
        self.assertIn("token_count", cols)

    def test_migrate_is_idempotent(self):
        # Re-running the migration on a DB that already has the column is a no-op.
        self.db._migrate()
        cols = {r[1] for r in self.db.conn.execute("PRAGMA table_info(lesson_cache)").fetchall()}
        self.assertIn("token_count", cols)

    def test_cache_stores_and_returns_token_count(self):
        self.db.store_cache("tok", "word", "B1", json.dumps({"lesson": "x"}), token_count=1234)
        self.assertEqual(self.db.get_cached("tok")["token_count"], 1234)

    def test_cache_token_count_defaults_to_zero(self):
        self.db.store_cache("tok0", "word", "B1", json.dumps({"lesson": "x"}))
        self.assertEqual(self.db.get_cached("tok0")["token_count"], 0)

    def test_cache_upsert_replaces_entry(self):
        self.db.store_cache("k", "word", "B1", json.dumps({"v": 1}))
        self.db.store_cache("k", "word", "B1", json.dumps({"v": 2}))
        self.assertEqual(self.db.get_cached("k")["v"], 2)

    def test_clear_cache_removes_all_entries(self):
        self.db.store_cache("a", "word", "B1", json.dumps({"v": 1}))
        self.db.store_cache("b", "word", "B1", json.dumps({"v": 2}))
        removed = self.db.clear_cache()
        self.assertEqual(removed, 2)
        self.assertIsNone(self.db.get_cached("a"))
        self.assertIsNone(self.db.get_cached("b"))

    # --- TTL expiration edges ------------------------------------------------

    def test_expired_entry_is_a_miss(self):
        old = datetime.now() - timedelta(days=31)
        self.db.store_cache("old", "word", "B1", json.dumps({"x": 1}), created_at=old)
        self.assertIsNone(self.db.get_cached("old", ttl_days=30))

    def test_fresh_entry_within_ttl_is_a_hit(self):
        recent = datetime.now() - timedelta(days=1)
        self.db.store_cache("recent", "word", "B1", json.dumps({"x": 2}), created_at=recent)
        self.assertEqual(self.db.get_cached("recent", ttl_days=30)["x"], 2)

    def test_ttl_is_configurable(self):
        two_days_ago = datetime.now() - timedelta(days=2)
        self.db.store_cache("k", "word", "B1", json.dumps({"x": 3}), created_at=two_days_ago)
        self.assertIsNone(self.db.get_cached("k", ttl_days=1))       # expired under 1-day TTL
        self.assertIsNotNone(self.db.get_cached("k", ttl_days=30))   # fresh under 30-day TTL


if __name__ == "__main__":
    unittest.main()
