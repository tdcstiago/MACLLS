# -*- coding: utf-8 -*-
"""
Test Suite for MACLLS Agent Prompts

This script verifies the correct import and content of the agent persona
prompts defined in agents/prompts.py.
"""

import os
import sys
import unittest

# --- Add project root to PYTHONPATH ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
# ---

# Import the prompts directly
try:
    from agents.prompts import (
        L1_SPECIALIST_PROMPT,
        L2_SPECIALIST_PROMPT,
        PEDAGOGUE_BRIDGE_PROMPT
    )
except ImportError as e:
    # If import fails, we still want to run tests to report it clearly
    print(f"ERROR: Could not import prompts. Check agents/prompts.py. {e}")
    L1_SPECIALIST_PROMPT = None
    L2_SPECIALIST_PROMPT = None
    PEDAGOGUE_BRIDGE_PROMPT = None


class TestAgentPrompts(unittest.TestCase):
    """
    Test cases for verifying the agent persona prompts.
    """

    def test_prompts_importable_and_defined(self):
        """
        Verifies that all three prompt constants can be successfully imported
        and are not None.
        """
        self.assertIsNotNone(L1_SPECIALIST_PROMPT, "L1_SPECIALIST_PROMPT should be imported and defined.")
        self.assertIsNotNone(L2_SPECIALIST_PROMPT, "L2_SPECIALIST_PROMPT should be imported and defined.")
        self.assertIsNotNone(PEDAGOGUE_BRIDGE_PROMPT, "PEDAGOGUE_BRIDGE_PROMPT should be imported and defined.")
        print("\n--- Prompts imported and defined successfully. ---")

    def test_prompts_are_non_empty_strings(self):
        """
        Verifies that all three prompt constants are non-empty strings.
        """
        self.assertIsInstance(L1_SPECIALIST_PROMPT, str, "L1_SPECIALIST_PROMPT should be a string.")
        self.assertGreater(len(L1_SPECIALIST_PROMPT), 0, "L1_SPECIALIST_PROMPT should not be empty.")
        
        self.assertIsInstance(L2_SPECIALIST_PROMPT, str, "L2_SPECIALIST_PROMPT should be a string.")
        self.assertGreater(len(L2_SPECIALIST_PROMPT), 0, "L2_SPECIALIST_PROMPT should not be empty.")
        
        self.assertIsInstance(PEDAGOGUE_BRIDGE_PROMPT, str, "PEDAGOGUE_BRIDGE_PROMPT should be a string.")
        self.assertGreater(len(PEDAGOGUE_BRIDGE_PROMPT), 0, "PEDAGOGUE_BRIDGE_PROMPT should not be empty.")
        print("--- All prompts are non-empty strings. ---")

    def test_pedagogue_bridge_contains_key_keywords(self):
        """
        Verifies that PEDAGOGUE_BRIDGE_PROMPT contains essential pedagogical keywords.
        """
        pedagogue_prompt_lower = PEDAGOGUE_BRIDGE_PROMPT.lower()
        
        # Check for 'mcp' (case-insensitive)
        self.assertIn('mcp', pedagogue_prompt_lower, "PEDAGOGUE_BRIDGE_PROMPT should mention 'MCP'.")
        
        # Check for 'similarity' (case-insensitive)
        self.assertIn('similarity', pedagogue_prompt_lower, "PEDAGOGUE_BRIDGE_PROMPT should mention 'similarity'.")
        
        # Check for 'cognate' (case-insensitive)
        self.assertIn('cognate', pedagogue_prompt_lower, "PEDAGOGUE_BRIDGE_PROMPT should mention 'cognate'.")
        
        print("--- PEDAGOGUE_BRIDGE_PROMPT contains key pedagogical keywords. ---")


if __name__ == '__main__':
    unittest.main()
