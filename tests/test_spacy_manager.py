import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import mcp_servers.spacy_manager as spacy_manager

# Whether a real spaCy model is installed here. When absent (CI / fresh checkout),
# the model-dependent tests skip and the fallback behaviour is exercised instead.
_probe, _ = spacy_manager.get_model("Portuguese")
MODEL_AVAILABLE = _probe is not None


class TestSpacyManager(unittest.TestCase):
    def setUp(self):
        spacy_manager.reset_for_tests()

    def tearDown(self):
        spacy_manager.reset_for_tests()

    # --- Fallback / negative paths (no real model needed) --------------------

    def test_unknown_language_returns_warning_and_negative_caches(self):
        # Patch spaCy non-None so we reach the "no model configured" branch even in
        # an environment where spaCy itself isn't installed.
        with mock.patch.object(spacy_manager, "spacy", object()):
            nlp, warn = spacy_manager.get_model("Klingon")
        self.assertIsNone(nlp)
        self.assertIn("No spaCy model configured", warn)
        # The negative result is cached so a missing model isn't retried per request.
        self.assertIn("Klingon", spacy_manager._warnings)

    def test_spacy_absent_returns_warning(self):
        with mock.patch.object(spacy_manager, "spacy", None):
            nlp, warn = spacy_manager.get_model("Portuguese")
        self.assertIsNone(nlp)
        self.assertIn("spaCy is not installed", warn)

    def test_reset_clears_caches(self):
        with mock.patch.object(spacy_manager, "_load", return_value=(object(), None)):
            spacy_manager.get_model("Portuguese")
        self.assertIn("Portuguese", spacy_manager._models)
        spacy_manager.reset_for_tests()
        self.assertNotIn("Portuguese", spacy_manager._models)

    # --- Singleton + thread-safety (mocked loader, no real model needed) -----

    def test_get_model_loads_once_and_shares_by_reference(self):
        sentinel = object()
        with mock.patch.object(spacy_manager, "_load", return_value=(sentinel, None)) as loader:
            a, _ = spacy_manager.get_model("Portuguese")
            b, _ = spacy_manager.get_model("Portuguese")
        self.assertIs(a, sentinel)
        self.assertIs(a, b)          # same object shared by reference
        loader.assert_called_once()  # loaded exactly once, then cached

    def test_concurrent_get_model_loads_exactly_once(self):
        # The crux: many threads racing on the same cold language must load it once
        # (double-checked locking). A small sleep widens the race window.
        calls = []
        calls_lock = threading.Lock()
        sentinel = object()

        def fake_load(lang):
            with calls_lock:
                calls.append(lang)
            time.sleep(0.02)
            return sentinel, None

        with mock.patch.object(spacy_manager, "_load", side_effect=fake_load):
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(spacy_manager.get_model, "Portuguese") for _ in range(8)]
                results = [f.result() for f in futures]

        self.assertEqual(len(calls), 1)                       # loaded once despite 8 racers
        self.assertTrue(all(nlp is sentinel for nlp, _ in results))

    # --- warmup() fail-safe --------------------------------------------------

    def test_warmup_success_returns_true(self):
        sentinel = object()
        with mock.patch.object(spacy_manager, "_load", return_value=(sentinel, None)):
            self.assertTrue(spacy_manager.warmup("Portuguese"))
        self.assertIs(spacy_manager.get_model("Portuguese")[0], sentinel)

    def test_warmup_missing_model_is_failsafe_and_warns(self):
        # Must not raise, must return False, must log a single warning — so a
        # mock / model-less deployment still starts normally.
        with self.assertLogs("mcp_servers.spacy_manager", level="WARNING") as cm:
            ok = spacy_manager.warmup("Klingon")
        self.assertFalse(ok)
        self.assertTrue(any("spacy.warmup_skipped" in line for line in cm.output))

    # --- Real-model behaviour (skips when no model installed) ----------------

    @unittest.skipUnless(MODEL_AVAILABLE, "requires an installed spaCy model")
    def test_singleton_returns_same_real_instance(self):
        a, _ = spacy_manager.get_model("Portuguese")
        b, _ = spacy_manager.get_model("Portuguese")
        self.assertIs(a, b)

    @unittest.skipUnless(MODEL_AVAILABLE, "requires an installed spaCy model")
    def test_ner_excluded_but_parse_pipeline_intact(self):
        nlp, warn = spacy_manager.get_model("Portuguese")
        self.assertIsNone(warn)
        self.assertNotIn("ner", nlp.pipe_names)   # excluded for RAM / latency
        self.assertIn("parser", nlp.pipe_names)    # dependency parse still available
        # POS + lemma must still be produced (tagger/morphologizer + lemmatizer kept).
        doc = nlp("O gato preto")
        self.assertTrue(any(tok.pos_ for tok in doc))
        self.assertTrue(all(tok.lemma_ is not None for tok in doc))


if __name__ == "__main__":
    unittest.main()
