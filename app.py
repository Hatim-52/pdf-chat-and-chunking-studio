import streamlit as st
import ollama
from ui.ui_components import load_css, render_header
from views.analysis_view import render_analysis_tab
from views.comparison_view import render_comparison_tab
from views.chat_view import render_chat_tab

# 1. Page Configuration & Theme
st.set_page_config(
    page_title="PDF Chat & Chunking Studio",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)
load_css("style.css")

# 2. Session State Initialization
for key, default in [
    ("pages", None), 
    ("chunks", None), 
    ("rag_engine", None), 
    ("messages", []), 
    ("pdf_name", ""), 
    ("last_processed_params", {})
]:
    if key not in st.session_state:
        st.session_state[key] = default

# 3. Sidebar Configuration
def render_sidebar() -> dict:
    st.sidebar.header("🛠️ System Configuration")
    provider = st.sidebar.selectbox(
        "Select RAG Provider",
        ["Local (TF-IDF)", "OpenAI", "Ollama"],
        help="Choose whether to use a local offline retriever, OpenAI APIs, or a local running Ollama instance."
    )
    
    api_key, ollama_model, ollama_url = None, "llama3", "http://localhost:11434"
    if provider == "OpenAI":
        api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Input your sk-... API key.")
        if not api_key:
            st.sidebar.warning("⚠️ Please provide your OpenAI API key to proceed.")
    elif provider == "Ollama":
        ollama_url = st.sidebar.text_input("Ollama Host URL", value="http://localhost:11434")
        try:
            models_info = ollama.list()
            ollama_models = [m['name'] for m in models_info.get('models', [])]
            ollama_model = st.sidebar.selectbox("Ollama Model", ollama_models) if ollama_models else "llama3"
        except Exception:
            st.sidebar.warning("⚠️ Could not connect to local Ollama (`ollama serve`).")
            ollama_model = st.sidebar.text_input("Ollama Model Name", value="llama3")

    st.sidebar.markdown("---")
    st.sidebar.header("📏 Chunking Settings")
    strategy = st.sidebar.selectbox(
        "Chunking Strategy",
        ["Recursive Character", "Semantic", "Sentence-based", "Fixed-size Character"]
    )
    chunk_size = st.sidebar.slider("Chunk Size (Characters)", 100, 2500, 800, 50)
    chunk_overlap = st.sidebar.slider("Chunk Overlap (Characters)", 0, 800, 150, 25)
    if chunk_overlap >= chunk_size:
        st.sidebar.error("Error: Overlap must be smaller than Chunk Size.")

    threshold_percentile = 80.0
    if strategy == "Semantic":
        threshold_percentile = st.sidebar.slider("Semantic Split Threshold (Percentile)", 10, 95, 80, 5)

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Advanced RAG Settings")
    with st.sidebar.expander("Configure Retrieval & Generation", expanded=False):
        top_k = st.slider("Top K Chunks to Retrieve", 1, 10, 3)
        similarity_threshold = st.slider("Similarity Score Threshold", 0.0, 1.0, 0.0, 0.05)
        openai_model = st.selectbox("OpenAI Chat Model", ["gpt-4o-mini", "gpt-4o"]) if provider == "OpenAI" else "gpt-4o-mini"
        system_prompt_template = st.text_area("Custom System Prompt", placeholder="Enter custom instructions with optional {context}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Quick Start Guide:**\n1. Choose **Local (TF-IDF)**\n2. Upload PDF in **Document Analysis**\n3. Click **Process & Index**\n4. Compare strategies or chat!")

    return {
        "provider": provider, "api_key": api_key, "ollama_url": ollama_url, "ollama_model": ollama_model,
        "strategy": strategy, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
        "threshold_percentile": threshold_percentile, "top_k": top_k, "similarity_threshold": similarity_threshold,
        "openai_model": openai_model, "system_prompt_template": system_prompt_template
    }

config = render_sidebar()

# 4. Main App Layout & Tabs
render_header()
tab1, tab2, tab3 = st.tabs([
    "📁 Document & Chunking Analysis", 
    "📊 Compare Strategies", 
    "💬 RAG Chat Studio"
])

with tab1:
    render_analysis_tab(config)
with tab2:
    render_comparison_tab()
with tab3:
    render_chat_tab(config)
