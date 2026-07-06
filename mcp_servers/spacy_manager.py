"""Singleton manager for spaCy pipelines.

Each language's spaCy model is loaded **once per process** and shared by reference
across every request, rather than reloaded per call. NER is excluded at load time
(we only consume POS / dependency / lemma), which lowers RAM and speeds up both the
load and per-document inference.

Thread-safety: models load lazily behind a double-checked lock, so two callers
racing on the same language (e.g. the L1/L2 specialists) can never double-load.

The public entry point, ``get_model(lang) -> (nlp, warning)``, preserves the exact
contract the linguistics server relied on, so the MCP layer above is untouched.
"""

import logging
import threading

logger = logging.getLogger(__name__)

# spaCy is optional: the module must import (and every non-spaCy code path must work)
# even when spaCy or a language model is absent. Callers degrade to an LLM-only path.
try:
    import spacy
except ImportError:  # pragma: no cover - environment-dependent
    spacy = None

# Display-name -> spaCy model. Single source of truth (linguistics_server imports
# this). Models are NOT pip dependencies; install on demand:
#   python -m spacy download pt_core_news_sm en_core_web_sm ...
SPACY_MODELS = {
    "English": "en_core_web_sm",
    "Portuguese": "pt_core_news_sm",
    "Spanish": "es_core_news_sm",
    "French": "fr_core_news_sm",
    "German": "de_core_news_sm",
    "Italian": "it_core_news_sm",
    "Romanian": "ro_core_news_sm",
}

# Components we never use are excluded at load time so they are never even built.
# The pipeline relies on POS (tagger/morphologizer), dependency parse (parser), and
# lemmas (lemmatizer); NER is dead weight here — excluding it is the RAM/latency win.
_EXCLUDED_COMPONENTS = ["ner"]

# lang -> loaded nlp pipeline (the shared singletons).
_models: dict = {}
# lang -> warning string for languages that could not be loaded. We cache the
# negative result too, so a missing model is not retried on every request.
_warnings: dict = {}
_lock = threading.Lock()


def get_model(lang: str):
    """Return ``(nlp, warning)`` for a display-language, loading once and caching.

    Thread-safe via double-checked locking: the hot path is an unlocked dict read;
    only a cache miss takes the lock, and the check is repeated inside it so two
    threads racing on the same language load it exactly once.

    Returns:
        ``(nlp, None)`` on success, or ``(None, warning)`` when spaCy or the model
        is unavailable — callers must degrade gracefully rather than fail.
    """
    # Fast path: already resolved (positive or negative), no lock required.
    if lang in _models:
        return _models[lang], None
    if lang in _warnings:
        return None, _warnings[lang]

    with _lock:
        # Re-check inside the lock: another thread may have loaded it while we waited.
        if lang in _models:
            return _models[lang], None
        if lang in _warnings:
            return None, _warnings[lang]

        nlp, warning = _load(lang)
        if nlp is not None:
            _models[lang] = nlp
        else:
            _warnings[lang] = warning
        return nlp, warning


def _load(lang: str):
    """Perform the actual spaCy load (called at most once per lang, under the lock)."""
    if spacy is None:
        return None, "spaCy is not installed; falling back to LLM-only sentence analysis."

    model_name = SPACY_MODELS.get(lang)
    if not model_name:
        return None, f"No spaCy model configured for '{lang}'; falling back to LLM-only analysis."

    try:
        nlp = spacy.load(model_name, exclude=_EXCLUDED_COMPONENTS)
    except Exception:  # model not downloaded / load failure
        return None, (
            f"spaCy model '{model_name}' is not installed "
            f"(run: python -m spacy download {model_name}); falling back to LLM-only analysis."
        )

    # Structured, correlation-id-carrying proof that the model cold-loads exactly
    # once per process — you will never see a second line for the same language.
    logger.info(
        "spacy.model_loaded",
        extra={
            "lang": lang,
            "model": model_name,
            "components": nlp.pipe_names,
            "excluded": _EXCLUDED_COMPONENTS,
            "cold_load": True,
        },
    )
    return nlp, None


def warmup(lang: str) -> bool:
    """Eagerly pre-load a language at startup to avoid first-request latency.

    Fail-safe: never raises. Returns True if a model is now resident, False if it
    could not be loaded (spaCy or the model absent). On failure it logs a single
    warning and returns — so mock / model-less deployments still start normally.
    """
    # Only warn on the first resolution; later warmup calls hit the negative cache
    # and must stay silent (Streamlit re-imports this module on every rerun).
    first_resolution = lang not in _models and lang not in _warnings
    nlp, warning = get_model(lang)
    if nlp is None:
        if first_resolution:
            logger.warning("spacy.warmup_skipped", extra={"lang": lang, "reason": warning})
        return False
    return True


def reset_for_tests() -> None:
    """Clear the singleton caches so a test can exercise cold-load behavior."""
    with _lock:
        _models.clear()
        _warnings.clear()
