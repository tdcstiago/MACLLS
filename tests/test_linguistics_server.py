import unittest
from mcp_servers.linguistics_server import (
    discover_contrastive_scenarios,
    analyze_sentence_structure,
    _scan_false_friends,
)
from mcp_servers.spacy_manager import get_model

# Whether a real spaCy model is available in this environment. When absent (CI /
# fresh checkout), the sentence tool must degrade gracefully rather than fail.
_nlp, _ = get_model("Portuguese")
MODEL_AVAILABLE = _nlp is not None


class TestWordMode(unittest.TestCase):

    def test_known_false_friend_returns_local_data(self):
        result = discover_contrastive_scenarios('pretender', 'English')
        self.assertEqual(result['dangerous']['word'], 'pretend')
        self.assertEqual(result['safe']['word'], 'intend')
        self.assertTrue(result['has_local_data'])
        self.assertIsNone(result['warning'])

    def test_common_word_defers_to_dynamic(self):
        result = discover_contrastive_scenarios('peixe', 'English')
        self.assertIsNone(result['dangerous'])
        self.assertFalse(result['has_local_data'])
        self.assertTrue(result['needs_dynamic_suggestions'])

    def test_empty_input(self):
        result = discover_contrastive_scenarios('', 'English')
        self.assertIsNotNone(result['warning'])
        self.assertFalse(result['has_local_data'])


class TestSentenceMode(unittest.TestCase):

    def test_scan_false_friends_direct(self):
        # Pure dictionary lookup — no spaCy needed.
        found = _scan_false_friends(["Quero", "pretender", "isso."], "English")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["l1_word"], "pretender")
        self.assertEqual(found[0]["dangerous"]["word"], "pretend")
        self.assertEqual(found[0]["safe"]["word"], "intend")

    def test_sentence_detects_embedded_false_friend(self):
        # Holds with or without a spaCy model (scan runs either way).
        result = analyze_sentence_structure("quero pretender isso", "Portuguese", "English")
        self.assertEqual(result["input_mode"], "sentence")
        detected = [f["l1_word"] for f in result["detected_false_friends"]]
        self.assertIn("pretender", detected)

    def test_sentence_no_false_friends(self):
        result = analyze_sentence_structure("a casa e grande", "Portuguese", "English")
        self.assertEqual(result["detected_false_friends"], [])

    @unittest.skipIf(MODEL_AVAILABLE, "spaCy model installed; fallback path not exercised")
    def test_graceful_degradation_without_model(self):
        # Models absent → whitespace tokens + false-friend scan, plus a warning.
        result = analyze_sentence_structure("quero pretender isso", "Portuguese", "English")
        self.assertFalse(result["has_local_data"])
        self.assertIsNotNone(result["warning"])
        self.assertEqual(len(result["tokens"]), 3)
        self.assertIsNone(result["tokens"][0]["pos"])  # no POS without a model

    def test_empty_sentence(self):
        result = analyze_sentence_structure("", "Portuguese", "English")
        self.assertIsNotNone(result["warning"])


if __name__ == '__main__':
    unittest.main()
