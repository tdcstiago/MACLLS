import unittest
from agents.orchestrator import LanguageOrchestrator, MAX_WORD_LEN, MAX_SENTENCE_LEN
from agents.prompts import lesson_labels


class TestInputClassification(unittest.TestCase):
    def test_classify_word_vs_sentence(self):
        self.assertEqual(LanguageOrchestrator._classify_input("pretender"), "word")
        self.assertEqual(LanguageOrchestrator._classify_input("guarda-chuva"), "word")
        self.assertEqual(LanguageOrchestrator._classify_input("   "), "word")  # empty → word
        self.assertEqual(LanguageOrchestrator._classify_input("eu pretendo viajar"), "sentence")

    def test_sanitize_input_caps(self):
        long = "x" * 1000
        self.assertEqual(len(LanguageOrchestrator._sanitize_input(long, "word")), MAX_WORD_LEN)
        self.assertEqual(len(LanguageOrchestrator._sanitize_input(long, "sentence")), MAX_SENTENCE_LEN)

    def test_normalize_level_defaults_to_b1(self):
        self.assertEqual(LanguageOrchestrator._normalize_level("A1"), "A1")
        self.assertEqual(LanguageOrchestrator._normalize_level("ZZ"), "B1")


class TestOrchestratorMock(unittest.TestCase):
    def setUp(self):
        # Instantiate without API key to trigger the secure mock fallback.
        self.orchestrator = LanguageOrchestrator(api_key=None)

    def test_mock_returns_twin_scenarios(self):
        result = self.orchestrator.process_lesson('pretender')
        self.assertEqual(result["input_mode"], "word")
        self.assertIn("MOCK ANALYSIS: API KEY MISSING", result["lesson"])
        self.assertEqual(result["dangerous"]["word"], "pretend")
        self.assertEqual(result["safe"]["word"], "intend")
        self.assertTrue(result["has_local_data"])

    def test_mock_lesson_uses_localized_labels(self):
        result = self.orchestrator.process_lesson('pretender')
        labels = lesson_labels('Portuguese')
        self.assertIn(labels["dangerous"], result["lesson"])
        self.assertIn(labels["safe"], result["lesson"])
        self.assertIn("pretend", result["lesson"])
        self.assertIn("intend", result["lesson"])

    def test_labels_switch_with_native_language(self):
        pt = self.orchestrator.process_lesson('pretender', l1_lang='Portuguese')
        en = self.orchestrator.process_lesson('pretender', l1_lang='English')
        self.assertEqual(pt["labels"], lesson_labels('Portuguese'))
        self.assertEqual(en["labels"], lesson_labels('English'))

    def test_mock_alternatives_empty_without_llm(self):
        result = self.orchestrator.process_lesson('pretender')
        self.assertEqual(result["alternatives"], [])

    def test_mock_common_word_has_no_false_friend(self):
        result = self.orchestrator.process_lesson('peixe')
        self.assertIsNone(result["dangerous"])
        self.assertIsNone(result["safe"])
        self.assertFalse(result["has_local_data"])
        self.assertIn("MOCK ANALYSIS", result["lesson"])

    def test_mock_result_carries_token_and_parse_flags(self):
        # New metadata: mock path has no LLM cost and never fails to parse.
        result = self.orchestrator.process_lesson('pretender')
        self.assertEqual(result["token_count"], 0)
        self.assertTrue(result["parse_ok"])


class TestProficiencyAndModeRouting(unittest.TestCase):
    def setUp(self):
        self.orchestrator = LanguageOrchestrator(api_key=None)

    def test_target_level_matches_user_config(self):
        for level in ("A1", "B2", "C2"):
            result = self.orchestrator.process_lesson('pretender', proficiency_level=level)
            self.assertEqual(result["target_level"], level)

    def test_target_level_defaults_to_b1(self):
        # Unset → B1; invalid → B1.
        self.assertEqual(self.orchestrator.process_lesson('pretender')["target_level"], "B1")
        self.assertEqual(
            self.orchestrator.process_lesson('pretender', proficiency_level="ZZ")["target_level"],
            "B1",
        )

    def test_word_input_routes_to_word_mode(self):
        result = self.orchestrator.process_lesson('pretender')
        self.assertEqual(result["input_mode"], "word")
        self.assertIn("dangerous", result)

    def test_sentence_input_routes_to_sentence_mode(self):
        result = self.orchestrator.process_lesson('quero pretender isso')
        self.assertEqual(result["input_mode"], "sentence")
        self.assertIn("l2_rendering", result)
        # Embedded curated false friend is surfaced even in the mock path.
        detected = [f["l1_word"] for f in result["detected_false_friends"]]
        self.assertIn("pretender", detected)


if __name__ == '__main__':
    unittest.main()
