import random

import streamlit as st
from agents.orchestrator import LanguageOrchestrator, PROMPT_VERSION
from database.db_manager import DatabaseManager
from mcp_servers.spacy_manager import warmup as warmup_spacy
from observability import configure_logging, set_correlation_id

# Structured JSON logging to stdout (Docker/OCI capture it). Idempotent across reruns.
configure_logging()


@st.cache_resource
def get_db() -> DatabaseManager:
    """Single shared DB connection across Streamlit reruns."""
    return DatabaseManager()


@st.cache_resource
def _warm_spacy(lang: str) -> bool:
    """Preload the L1 spaCy model once per server so the first sentence analysis
    isn't slow. Cached per language; warmup() is fail-safe and never raises."""
    return warmup_spacy(lang)


db = get_db()

# Warm the persisted native language so a demo's first sentence parse is instant.
_warm_spacy(db.get_setting("l1_lang", "Portuguese"))

# --- Page Configuration ---
st.set_page_config(
    page_title="MACLLS - Language Learning System",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- App Title and Description ---
st.title("🔍 MACLLS: Multi-Agent System for Contrastive Linguistics")
st.markdown("""
Welcome to the presentation layer for our Multi-Agent System for Language Learning.
This interface allows you to compare words across languages to understand their similarities and differences, 
powered by a sophisticated backend of linguistic analysis agents.
""")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Languages come from the DB registry, not a hardcoded list.
    languages = [row["display_name"] for row in db.get_active_languages()]

    def _index_of(options, saved, fallback):
        if saved in options:
            return options.index(saved)
        return options.index(fallback) if fallback in options else 0

    l1_selection = st.selectbox(
        "Select Native Language (L1)", languages,
        index=_index_of(languages, db.get_setting("l1_lang", "Portuguese"), "Portuguese"),
    )
    l2_selection = st.selectbox(
        "Select Target Language (L2)", languages,
        index=_index_of(languages, db.get_setting("l2_lang", "English"), "English"),
    )

    cefr_levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    proficiency_level = st.selectbox(
        "CEFR Proficiency Level", cefr_levels,
        index=_index_of(cefr_levels, db.get_setting("cefr_level", "B1"), "B1"),
    )

    # Persist the user's selections so they survive an app restart.
    db.set_setting("l1_lang", l1_selection)
    db.set_setting("l2_lang", l2_selection)
    db.set_setting("cefr_level", proficiency_level)

    st.markdown("---")
    
    # Tenta buscar do secrets.toml automaticamente, se não achar fica em branco
    PLACEHOLDER_KEY = "PASTE_YOUR_ROTATED_KEY_HERE"
    default_key = ""
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
        # Ignora o placeholder do template para preservar a mensagem "API Key is missing"
        if secret_key and secret_key != PLACEHOLDER_KEY:
            default_key = secret_key
    except Exception:
        pass

    api_key_input = st.text_input(
        "Google Gemini API Key",
        type="password",
        value=default_key,
        help="Required for AI-powered analysis."
    )

    st.markdown("---")
    if st.button("🧹 Limpar cache de lições", help="Remove todas as respostas em cache."):
        removed = db.clear_cache()
        st.success(f"Cache limpo ({removed} entrada(s) removida(s)).")

def build_card_texts(input_text: str, result: dict):
    """Derive a flashcard front/back from an analysis result dict."""
    front = f"Analise: {input_text.strip()}"
    parts = []
    if result["input_mode"] == "word":
        safe = result.get("safe")
        if safe:
            parts.append(f"✅ {safe['word']}" + (f" — {safe['meaning']}" if safe.get("meaning") else ""))
        dangerous = result.get("dangerous")
        if dangerous:
            parts.append(f"⚠️ Falso amigo: {dangerous['word']} — {dangerous['meaning']}")
        alts = result.get("alternatives") or []
        if alts:
            parts.append("Relacionadas: " + ", ".join(alts[:5]))
    else:
        if result.get("l2_rendering"):
            parts.append(f"✅ {result['l2_rendering']}")
        notes = result.get("structural_notes") or []
        if notes:
            parts.append("• " + notes[0])
    return front, ("\n\n".join(parts) if parts else "(sem dados)")


tab_analysis, tab_practice = st.tabs(["🔍 Contrastive Analysis", "🧠 Daily Practice"])

# --- Tab 1: Hybrid (word or sentence) Analysis ---
with tab_analysis:
    st.header("📖 Contrastive Analysis")

    with st.form(key="analysis_form"):
        input_l1 = st.text_area(
            "Digite uma palavra ou frase para analisar",
            placeholder="ex.: Pretender  •  ex.: Eu pretendo assistir o filme",
            height=100,
        )
        submit_button = st.form_submit_button(label="Analisar")

    if submit_button:
        if not input_l1.strip():
            st.warning("Por favor, digite uma palavra ou frase para analisar.")
        elif l1_selection == l2_selection:
            st.warning("Native (L1) and Target (L2) languages must be different.")
        else:
            final_key = api_key_input if api_key_input else default_key
            if not final_key:
                st.error("🔑 API Key is missing! Please paste it in the sidebar or configure secrets.toml.")
            else:
                with st.spinner("Our agent-led team is analyzing your input..."):
                    try:
                        # One correlation id per analysis request, traced across all agents.
                        set_correlation_id()
                        orchestrator = LanguageOrchestrator(api_key=final_key, db=db)
                        result = orchestrator.process_lesson(
                            input_l1=input_l1,
                            l1_lang=l1_selection,
                            l2_lang=l2_selection,
                            proficiency_level=proficiency_level,
                        )
                        # Persist so the "Save" button (a rerun) still has the data.
                        st.session_state["analysis"] = {
                            "result": result, "input": input_l1,
                            "l1": l1_selection, "l2": l2_selection,
                        }
                    except Exception as e:
                        st.session_state.pop("analysis", None)
                        st.error(f"An error occurred during analysis: {e}")

    # Render the most recent successful analysis (survives button reruns).
    analysis = st.session_state.get("analysis")
    if analysis:
        result = analysis["result"]
        labels = result["labels"]

        # The LLM's JSON payload failed to parse (fallback path) → warn the user.
        if not result.get("parse_ok", True):
            st.warning(
                "⚠️ A IA gerou uma resposta incompleta ou mal formatada. "
                "Por favor, tente analisar novamente."
            )

        st.success("Analysis Complete!")
        st.info(f"🎯 Calibrado para {result['target_level']}")

        if result["input_mode"] == "word":
            dangerous, safe = result["dangerous"], result["safe"]
            if dangerous or safe:
                col_danger, col_safe = st.columns(2)
                with col_danger:
                    if dangerous:
                        st.error(
                            f"**{labels['dangerous']}**\n\n"
                            f"**{dangerous['word']}**\n\n{dangerous['meaning']}"
                        )
                with col_safe:
                    if safe:
                        st.success(
                            f"**{labels['safe']}**\n\n"
                            f"**{safe['word']}**\n\n{safe['meaning']}"
                        )
            elif result["warning"]:
                st.info(result["warning"])
        else:
            if result.get("l2_rendering"):
                st.markdown(f"**✅ Idiomatic {analysis['l2']} rendering:**")
                st.success(result["l2_rendering"])
            notes = result.get("structural_notes") or []
            if notes:
                st.markdown("**🔧 Structural transfer notes:**")
                for note in notes:
                    st.markdown(f"- {note}")
            detected = result.get("detected_false_friends") or []
            if detected:
                st.markdown("**⚠️ Embedded false friends:**")
                for ff in detected:
                    st.warning(
                        f"**{ff['l1_word']}** → dangerous *{ff['dangerous']['word']}* "
                        f"({ff['dangerous']['meaning']}); safe *{ff['safe']['word']}* "
                        f"({ff['safe']['meaning']})"
                    )

        st.header("📜 Pedagogical Output")
        st.markdown(result["lesson"])

        alternatives = result.get("alternatives") or []
        if alternatives:
            title = ("🔎 Opções similares encontradas"
                     if result["input_mode"] == "word"
                     else "🔎 Frasings alternativos")
            with st.expander(title):
                for word in alternatives:
                    st.markdown(f"- **{word}**")

        # Save the current analysis as a spaced-repetition flashcard.
        if st.button("💾 Save to Daily Practice"):
            front, back = build_card_texts(analysis["input"], result)
            cache_key = DatabaseManager.build_cache_key(
                analysis["l1"], analysis["l2"], result["target_level"],
                result["input_mode"], analysis["input"], PROMPT_VERSION,
            )
            db.add_flashcard(cache_key, front, back)
            st.success("Salvo em Daily Practice! 🧠")

# --- Tab 2: Daily Practice (SRS flashcards) ---
with tab_practice:
    st.header("🧠 Daily Practice")

    SESSION_KEYS = ("review_queue", "total_due_today", "reviewed", "srs_show_answer")

    def _clear_session():
        for key in SESSION_KEYS:
            st.session_state.pop(key, None)

    if "review_queue" not in st.session_state:
        # No active session → offer to start one (or report nothing due).
        due_now = db.get_due_flashcards()
        if not due_now:
            st.success("Nenhum cartão para revisar agora. 🎉 Salve análises na aba anterior.")
        else:
            st.write(f"Você tem **{len(due_now)}** cartão(ões) para revisar hoje.")
            if st.button("▶️ Start Review Session"):
                random.shuffle(due_now)
                st.session_state.review_queue = due_now
                st.session_state.total_due_today = len(due_now)
                st.session_state.reviewed = 0
                st.session_state.srs_show_answer = False
                st.rerun()
    else:
        queue = st.session_state.review_queue
        total = st.session_state.total_due_today
        reviewed = st.session_state.reviewed

        if not queue:
            # Session finished: celebrate, then clear so a new one can start.
            st.success("🎉 All caught up for today!")
            st.caption(f"{reviewed} cartão(ões) revisado(s) nesta sessão.")
            _clear_session()
            if st.button("↺ New session"):
                st.rerun()
        else:
            # Progress tracker.
            st.progress(reviewed / total if total else 0.0)
            st.caption(f"Cards Reviewed: {reviewed} | Remaining: {len(queue)}")

            card = queue[0]  # always review the head of the queue
            st.markdown(f"### {card['front_text']}")

            if not st.session_state.get("srs_show_answer", False):
                if st.button("Show Answer"):
                    st.session_state.srs_show_answer = True
                    st.rerun()
            else:
                st.markdown("---")
                st.markdown(card["back_text"])
                st.write("")
                grade_buttons = [("Again", 0), ("Hard", 1), ("Good", 2), ("Easy", 3)]
                for col, (label, grade) in zip(st.columns(4), grade_buttons):
                    if col.button(label, key=f"srs_grade_{grade}", use_container_width=True):
                        # Defensive: a rapid double-click could fire this handler after
                        # the queue was already drained — verify before popping.
                        if st.session_state.review_queue:
                            db.update_flashcard_progress(card["id"], grade)
                            current = st.session_state.review_queue.pop(0)
                            if grade == 0:  # Again → requeue so it reappears this session
                                st.session_state.review_queue.append(current)
                            else:
                                st.session_state.reviewed += 1
                        st.session_state.srs_show_answer = False
                        st.rerun()

st.markdown("---")
st.info("This application is the final phase of the MACLLS project, demonstrating the system's presentation layer.")