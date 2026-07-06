import contextvars
import json
import logging
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor

from observability.correlation import (
    NO_CORRELATION_ID,
    CorrelationIdFilter,
    correlation_scope,
    get_correlation_id,
    new_correlation_id,
    set_correlation_id,
)
from observability.logger_setup import (
    JsonFormatter,
    configure_logging,
    reset_logging_for_tests,
)


class TestCorrelationId(unittest.TestCase):
    def tearDown(self):
        set_correlation_id(NO_CORRELATION_ID)  # avoid leaking id between tests

    def test_new_id_is_unique_hex(self):
        a, b = new_correlation_id(), new_correlation_id()
        self.assertNotEqual(a, b)
        self.assertEqual(len(a), 32)
        int(a, 16)  # raises if not valid hex

    def test_set_and_get(self):
        self.assertEqual(set_correlation_id("abc123"), "abc123")
        self.assertEqual(get_correlation_id(), "abc123")

    def test_set_generates_when_omitted(self):
        cid = set_correlation_id()
        self.assertEqual(get_correlation_id(), cid)
        self.assertNotEqual(cid, NO_CORRELATION_ID)

    def test_scope_sets_and_restores(self):
        set_correlation_id("outer")
        with correlation_scope("inner") as cid:
            self.assertEqual(cid, "inner")
            self.assertEqual(get_correlation_id(), "inner")
        self.assertEqual(get_correlation_id(), "outer")

    def test_copy_context_propagates_to_thread(self):
        # This is the ThreadPoolExecutor fix: a copied context carries the id into
        # the worker, while a naive submission does not.
        cid = set_correlation_id("thread-test")
        with ThreadPoolExecutor(max_workers=1) as pool:
            with_copy = pool.submit(contextvars.copy_context().run, get_correlation_id).result()
            naive = pool.submit(get_correlation_id).result()
        self.assertEqual(with_copy, cid)
        self.assertEqual(naive, NO_CORRELATION_ID)


class TestJsonFormatter(unittest.TestCase):
    def _record(self, level=logging.INFO, msg="agent.done", exc_info=None, **extra):
        rec = logging.LogRecord("agents.orchestrator", level, "o.py", 1, msg, None, exc_info)
        for key, value in extra.items():
            setattr(rec, key, value)
        return rec

    def test_emits_valid_json_with_core_keys(self):
        rec = self._record(correlation_id="cid1", agent="L1", duration_ms=812, token_count=1450)
        obj = json.loads(JsonFormatter().format(rec))
        self.assertEqual(obj["level"], "INFO")
        self.assertEqual(obj["logger"], "agents.orchestrator")
        self.assertEqual(obj["message"], "agent.done")
        self.assertEqual(obj["correlation_id"], "cid1")
        self.assertEqual(obj["agent"], "L1")
        self.assertEqual(obj["duration_ms"], 812)
        self.assertEqual(obj["token_count"], 1450)
        self.assertIn("timestamp", obj)

    def test_falls_back_to_live_id_without_stamp(self):
        set_correlation_id("live-cid")
        obj = json.loads(JsonFormatter().format(self._record()))  # no correlation_id attr
        self.assertEqual(obj["correlation_id"], "live-cid")
        set_correlation_id(NO_CORRELATION_ID)

    def test_exception_block(self):
        try:
            raise ValueError("boom")
        except ValueError:
            rec = self._record(level=logging.ERROR, msg="err", exc_info=sys.exc_info(),
                               correlation_id="-")
        obj = json.loads(JsonFormatter().format(rec))
        self.assertIn("error", obj)
        self.assertIn("boom", obj["error"])


class TestCorrelationIdFilter(unittest.TestCase):
    def test_filter_stamps_current_id(self):
        set_correlation_id("filter-cid")
        rec = logging.LogRecord("x", logging.INFO, "o.py", 1, "m", None, None)
        self.assertTrue(CorrelationIdFilter().filter(rec))
        self.assertEqual(rec.correlation_id, "filter-cid")
        set_correlation_id(NO_CORRELATION_ID)


class TestConfigureLogging(unittest.TestCase):
    def setUp(self):
        reset_logging_for_tests()

    def tearDown(self):
        reset_logging_for_tests()

    def test_idempotent_single_handler(self):
        configure_logging()
        configure_logging()
        configure_logging()
        self.assertEqual(len(logging.getLogger().handlers), 1)

    def test_level_from_argument(self):
        configure_logging("DEBUG")
        self.assertEqual(logging.getLogger().level, logging.DEBUG)

    def test_handler_uses_json_formatter(self):
        configure_logging()
        handler = logging.getLogger().handlers[0]
        self.assertIsInstance(handler.formatter, JsonFormatter)


if __name__ == "__main__":
    unittest.main()
