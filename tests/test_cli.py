import os
import unittest
from unittest import mock

import cli


class TestCliKeyResolution(unittest.TestCase):
    """Network-free tests for the CLI's API-key resolution logic."""

    def test_env_var_takes_precedence(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "env-key-123"}):
            self.assertEqual(cli.load_api_key(), "env-key-123")

    def test_placeholder_is_treated_as_missing(self):
        # A placeholder key must never be used → resolves to None (mock mode).
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": cli.PLACEHOLDER_KEY}):
            self.assertIsNone(cli.load_api_key())


if __name__ == "__main__":
    unittest.main()
