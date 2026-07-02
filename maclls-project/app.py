import streamlit as st
from agents.orchestrator import LanguageOrchestrator
from database.db_manager import DatabaseManager


@st.cache_resource
def get_db() -> DatabaseManager:
    """Single shared DB connection across Streamlit reruns."""
    return DatabaseManager()


db = get_db()

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

# --- Main Content Area: Hybrid (word or sentence) Analysis ---
st.header("📖 Contrastive Analysis")

with st.form(key="analysis_form"):
    input_l1 = st.text_area(
        "Digite uma palavra ou frase para analisar",
        placeholder="ex.: Pretender  •  ex.: Eu pretendo assistir o filme",
        height=100,
    )
    submit_button = st.form_submit_button(label="Analisar")

# --- Processing and Displaying Results ---
if submit_button:
    if not input_l1.strip():
        st.warning("Por favor, digite uma palavra ou frase para analisar.")
    elif l1_selection == l2_selection:
        st.warning("Native (L1) and Target (L2) languages must be different.")
    else:
        # Prioritiza a chave digitada na tela; se vazia, usa a do segredo
        final_key = api_key_input if api_key_input else default_key

        if not final_key:
            st.error("🔑 API Key is missing! Please paste it in the sidebar or configure secrets.toml.")
        else:
            with st.spinner("Our agent-led team is analyzing your input..."):
                try:
                    # 1. Instancia o orquestrador com a chave e o cache local
                    orchestrator = LanguageOrchestrator(api_key=final_key, db=db)

                    # 2. Classificação (palavra/frase) + pipeline multi-agente adaptativo
                    result = orchestrator.process_lesson(
                        input_l1=input_l1,
                        l1_lang=l1_selection,       # Envia ex: "Portuguese"
                        l2_lang=l2_selection,       # Envia ex: "English"
                        proficiency_level=proficiency_level,
                    )

                    st.success("Analysis Complete!")
                    # Badge de calibração CEFR
                    st.info(f"🎯 Calibrado para {result['target_level']}")

                    labels = result["labels"]

                    if result["input_mode"] == "word":
                        # --- WORD MODE: cartões Seguro vs. Perigoso ---
                        dangerous = result["dangerous"]
                        safe = result["safe"]
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
                        # --- SENTENCE MODE: rendering idiomático + notas estruturais ---
                        if result.get("l2_rendering"):
                            st.markdown(f"**✅ Idiomatic {l2_selection} rendering:**")
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

                    # Lição pedagógica completa dos agentes
                    st.header("📜 Pedagogical Output")
                    st.markdown(result["lesson"])

                    # Seção interativa: opções similares / frasings alternativos
                    alternatives = result.get("alternatives") or []
                    if alternatives:
                        title = ("🔎 Opções similares encontradas"
                                 if result["input_mode"] == "word"
                                 else "🔎 Frasings alternativos")
                        with st.expander(title):
                            for word in alternatives:
                                st.markdown(f"- **{word}**")

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")

st.markdown("---")
st.info("This application is the final phase of the MACLLS project, demonstrating the system's presentation layer.")