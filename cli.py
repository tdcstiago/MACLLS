"""MACLLS Agents CLI — run the multi-agent contrastive analysis from the terminal.

Examples:
    python cli.py "pretender"
    python cli.py "Eu pretendo viajar" --l1 Portuguese --l2 English --cefr C1
"""

import argparse
import os
import re
import sys
from pathlib import Path

from agents.orchestrator import LanguageOrchestrator
from database.db_manager import DatabaseManager

PLACEHOLDER_KEY = "PASTE_YOUR_ROTATED_KEY_HERE"
SECRETS_PATH = Path(".streamlit/secrets.toml")


def load_api_key() -> str | None:
    """Resolve the Gemini key from the environment, else .streamlit/secrets.toml.

    Uses tomllib when available and falls back to a simple regex so the CLI works
    even without Streamlit loaded. The template placeholder is treated as missing."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key and SECRETS_PATH.exists():
        text = SECRETS_PATH.read_text(encoding="utf-8")
        try:
            import tomllib  # Python 3.11+
            key = tomllib.loads(text).get("GEMINI_API_KEY")
        except Exception:
            match = re.search(r'GEMINI_API_KEY\s*=\s*"([^"]+)"', text)
            key = match.group(1) if match else None
    if key and key.strip() and key != PLACEHOLDER_KEY:
        return key.strip()
    return None


def _print_result(result: dict) -> None:
    print(f"\n🎯 CEFR: {result['target_level']}  ({result['input_mode']} mode)\n")

    if not result.get("parse_ok", True):
        print("⚠️  The AI produced an incomplete/malformed response — try again.\n")

    if result["input_mode"] == "word":
        dangerous, safe = result.get("dangerous"), result.get("safe")
        if dangerous:
            print(f"⚠️  Dangerous (false friend): {dangerous['word']}"
                  + (f" — {dangerous['meaning']}" if dangerous.get("meaning") else ""))
        if safe:
            print(f"✅  Safe (correct target):   {safe['word']}"
                  + (f" — {safe['meaning']}" if safe.get("meaning") else ""))
        alts = result.get("alternatives") or []
        if alts:
            print(f"🔎  Related: {', '.join(alts)}")
    else:
        if result.get("l2_rendering"):
            print(f"✅  Idiomatic rendering: {result['l2_rendering']}")
        for note in result.get("structural_notes") or []:
            print(f"🔧  {note}")

    print("\n" + "-" * 60)
    print(result["lesson"])
    print("-" * 60)

    tokens = result.get("token_count")
    if tokens:
        print(f"\n(prompt tokens: {tokens})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="MACLLS — multi-agent contrastive language analysis (word or sentence).",
    )
    parser.add_argument("text", help="The word or sentence to analyze (in L1).")
    parser.add_argument("--l1", default="Portuguese", help="Native language (default: Portuguese).")
    parser.add_argument("--l2", default="English", help="Target language (default: English).")
    parser.add_argument("--cefr", default="B1", help="CEFR level A1–C2 (default: B1).")
    args = parser.parse_args(argv)

    # Lessons (and our labels) contain emoji; force UTF-8 so Windows' cp1252
    # console doesn't crash with a UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    api_key = load_api_key()
    if not api_key:
        print("🔑 No GEMINI_API_KEY found (env or .streamlit/secrets.toml) — running in MOCK mode.\n",
              file=sys.stderr)

    db = DatabaseManager()  # reuses the same SQLite cache/flashcards as the web app
    try:
        orchestrator = LanguageOrchestrator(api_key=api_key, db=db)
        result = orchestrator.process_lesson(
            args.text, l1_lang=args.l1, l2_lang=args.l2, proficiency_level=args.cefr
        )
        _print_result(result)
        return 0
    except Exception as e:  # noqa: BLE001 - surface a clean CLI error
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
