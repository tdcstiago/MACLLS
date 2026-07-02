import json
import types
import unittest

from agents.orchestrator import LanguageOrchestrator
from google.genai.errors import APIError


def make_api_error(code):
    """Build an APIError carrying a status code without invoking its
    response-parsing __init__ (which expects a live HTTP response object)."""
    err = APIError.__new__(APIError)
    err.code = code
    err.message = f"error {code}"
    return err


def make_response(text):
    cand = types.SimpleNamespace(finish_reason="STOP")
    return types.SimpleNamespace(text=text, candidates=[cand])


def pedagogue_json(lesson, options, safe="", dangerous="none"):
    return json.dumps(
        {
            "lesson_markdown": lesson,
            "safe_target": safe,
            "dangerous_target": dangerous,
            "similar_options": list(options),
        }
    )


class ScriptedModels:
    """Fake `client.models` that plays back a scripted sequence of results/errors
    and records the models it was called with."""

    def __init__(self, script):
        self._script = script  # callable(model, contents) -> response | raises
        self.calls = []

    def generate_content(self, model, contents, config=None):
        self.calls.append(model)
        return self._script(model, contents)


def fake_client(script):
    return types.SimpleNamespace(models=ScriptedModels(script))


class TestRetryLogic(unittest.TestCase):
    def setUp(self):
        self.orch = LanguageOrchestrator(api_key="x")

    def test_retries_transient_then_succeeds(self):
        seq = [make_api_error(503), make_api_error(503), make_response("ok")]

        def script(model, contents):
            item = seq.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

        self.orch.client = fake_client(script)
        out = self.orch._generate_content_with_retry("p", retries=3, delay=0)
        self.assertEqual(out, "ok")
        self.assertEqual(len(self.orch.client.models.calls), 3)

    def test_non_retryable_raises_immediately(self):
        def script(model, contents):
            raise make_api_error(400)

        self.orch.client = fake_client(script)
        with self.assertRaises(APIError):
            self.orch._generate_content_with_retry("p", retries=3, delay=0)
        self.assertEqual(len(self.orch.client.models.calls), 1)

    def test_falls_back_to_secondary_model(self):
        def script(model, contents):
            if model == "primary":
                raise make_api_error(503)
            return make_response("from-fallback")

        self.orch.client = fake_client(script)
        out = self.orch._generate_content_with_retry(
            "p", primary_model="primary", fallback_model="fallback", retries=3, delay=0
        )
        self.assertEqual(out, "from-fallback")
        calls = self.orch.client.models.calls
        self.assertEqual(calls.count("primary"), 3)
        self.assertEqual(calls.count("fallback"), 1)

    def test_empty_response_raises_with_reason(self):
        def script(model, contents):
            return make_response(None)

        self.orch.client = fake_client(script)
        with self.assertRaises(RuntimeError) as ctx:
            self.orch._generate_content_with_retry("p", retries=1, delay=0)
        self.assertIn("finish_reason", str(ctx.exception))


class TestPayloadParsing(unittest.TestCase):
    def test_parses_json_payload(self):
        text = pedagogue_json("# Lição", ["intend", "contender"], safe="intend", dangerous="pretend")
        parsed = LanguageOrchestrator._parse_pedagogue_payload(text)
        self.assertEqual(parsed["lesson"], "# Lição")
        self.assertEqual(parsed["safe_target"], "intend")
        self.assertEqual(parsed["dangerous_target"], "pretend")
        self.assertEqual(parsed["similar_options"], ["intend", "contender"])

    def test_strips_code_fences(self):
        text = "```json\n" + pedagogue_json("# L", ["fish"], safe="fish") + "\n```"
        parsed = LanguageOrchestrator._parse_pedagogue_payload(text)
        self.assertEqual(parsed["lesson"], "# L")
        self.assertEqual(parsed["similar_options"], ["fish"])

    def test_invalid_json_falls_back_to_raw_lesson(self):
        parsed = LanguageOrchestrator._parse_pedagogue_payload("not json at all")
        self.assertEqual(parsed["lesson"], "not json at all")
        self.assertEqual(parsed["similar_options"], [])


class TestParallelPipeline(unittest.TestCase):
    def setUp(self):
        self.orch = LanguageOrchestrator(api_key="x")

    def test_false_friend_pipeline_generates_structural_options(self):
        def script(model, contents):
            if "interference risk" in contents:
                return make_response("[L1]")
            if "native model" in contents:
                return make_response("[L2]")
            return make_response(
                pedagogue_json(
                    "# Lesson",
                    ["intend", "contender", "intention", "pretense", "contend"],
                    safe="intend", dangerous="pretend",
                )
            )

        self.orch.client = fake_client(script)
        result = self.orch.process_lesson("pretender", "Portuguese", "English")
        self.assertEqual(result["lesson"], "# Lesson")
        # Curated data wins for the cards on a known false friend.
        self.assertEqual(result["dangerous"]["word"], "pretend")
        self.assertEqual(result["safe"]["word"], "intend")
        # similar_options is now a plain list of strings.
        self.assertEqual(
            result["alternatives"],
            ["intend", "contender", "intention", "pretense", "contend"],
        )

    def test_common_word_pipeline_generates_semantic_field(self):
        # 'peixe' has no false friend → semantic-field suggestions; cards come from
        # the LLM's safe_target/dangerous_target.
        captured = {}

        def script(model, contents):
            if "similar_options" in contents:  # the pedagogue (structured) call
                captured["pedagogue_prompt"] = contents
                return make_response(
                    pedagogue_json(
                        "# Peixe",
                        ["fish", "fishing", "fisherman", "seafood", "salmon"],
                        safe="fish", dangerous="none",
                    )
                )
            return make_response("[specialist]")

        self.orch.client = fake_client(script)
        result = self.orch.process_lesson("peixe", "Portuguese", "English")
        self.assertFalse(result["has_local_data"])
        self.assertIsNone(result["dangerous"])          # dangerous_target == "none"
        self.assertEqual(result["safe"]["word"], "fish")  # from LLM safe_target
        self.assertEqual(
            result["alternatives"],
            ["fish", "fishing", "fisherman", "seafood", "salmon"],
        )
        # The pedagogue was told to use the semantic/lexical field, not lookalikes.
        self.assertIn("SEMANTIC and LEXICAL field", captured["pedagogue_prompt"])

    def test_plain_word_lesson_prompt_mandates_rich_sections(self):
        # Regression: a plain word must produce a scaffolded, multi-section lesson,
        # not a one-liner. B1 → at least 2 example sentences.
        captured = {}

        def script(model, contents):
            if "similar_options" in contents:
                captured["prompt"] = contents
                return make_response(pedagogue_json("# Peixe", ["fish"], safe="fish"))
            return make_response("[specialist]")

        self.orch.client = fake_client(script)
        self.orch.process_lesson("peixe", "Portuguese", "English", "B1")
        prompt = captured["prompt"]
        self.assertIn("Uso em Contexto", prompt)              # mandated section
        self.assertIn("Nuances & Registro", prompt)           # mandated section
        self.assertIn("at least 2 example", prompt)           # max_examples for B1
        self.assertIn("never a single sentence", prompt)      # anti-terseness guard

    def test_agent_failure_propagates_as_runtimeerror(self):
        def script(model, contents):
            if "interference risk" in contents:
                raise make_api_error(400)
            return make_response("[L2]")

        self.orch.client = fake_client(script)
        with self.assertRaises(RuntimeError):
            self.orch.process_lesson("pretender", "Portuguese", "English")


if __name__ == "__main__":
    unittest.main()
