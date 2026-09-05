import streamlit as st
import pandas as pd
import altair as alt
import re
from core.chunker import chunk_document, RecursiveCharacterTextSplitter, SemanticTextSplitter
from ui.ui_components import render_comparison_metric_card, render_boundary_highlights

def _split_sample(sample_text: str, params: dict) -> list[str]:
    """Helper to split a small sample text for visual boundary inspection."""
    strat = params["strat"]
    size = params["size"]
    overlap = params["overlap"]
    thresh = params["thresh"]
    
    if strat == "Recursive Character":
        return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_text(sample_text)
    elif strat == "Semantic":
        return SemanticTextSplitter(chunk_size=size, chunk_overlap=overlap, threshold_percentile=thresh).split_text(sample_text)
    elif strat == "Sentence-based":
        sentence_regex = re.compile(r'(?<=[.!?])\s+')
        sents = [s.strip() for s in sentence_regex.split(sample_text) if s.strip()]
        splits, curr, curr_len = [], [], 0
        for s in sents:
            if curr_len + len(s) > size and curr:
                splits.append(" ".join(curr))
                curr, curr_len = [], 0
            curr.append(s)
            curr_len += len(s) + 1
        if curr:
            splits.append(" ".join(curr))
        return splits
    else:  # Fixed Character
        return [sample_text[i:i + size] for i in range(0, len(sample_text), max(1, size - overlap))]

def render_comparison_tab():
    """Renders Tab 2: Strategy Comparison Studio."""
    st.subheader("📊 Chunking Strategy Comparison")
    
    if not st.session_state.pages:
        st.info("💡 Please upload a document in the 'Document & Chunking Analysis' tab first to enable comparison.")
        return
        
    st.write("Compare different chunking strategies and parameter configurations on the same uploaded document.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔵 Configuration A")
        strat_a = st.selectbox("Strategy A", ["Recursive Character", "Semantic", "Sentence-based", "Fixed-size Character"], key="strat_a", index=0)
        size_a = st.slider("Chunk Size A", 100, 2500, 800, 50, key="size_a")
        overlap_a = st.slider("Chunk Overlap A", 0, 800, 150, 25, key="overlap_a")
        thresh_a = st.slider("Semantic Split Percentile A", 10, 95, 80, key="thresh_a") if strat_a == "Semantic" else 80.0
            
    with col2:
        st.markdown("### 🟢 Configuration B")
        strat_b = st.selectbox("Strategy B", ["Recursive Character", "Semantic", "Sentence-based", "Fixed-size Character"], key="strat_b", index=1)
        size_b = st.slider("Chunk Size B", 100, 2500, 600, 50, key="size_b")
        overlap_b = st.slider("Chunk Overlap B", 0, 800, 100, 25, key="overlap_b")
        thresh_b = st.slider("Semantic Split Percentile B", 10, 95, 80, key="thresh_b") if strat_b == "Semantic" else 80.0
            
    if st.button("⚡ Run Comparison", type="primary", use_container_width=True):
        with st.spinner("Processing document under both configurations..."):
            chunks_a = chunk_document(st.session_state.pages, strat_a, size_a, overlap_a, threshold_percentile=thresh_a)
            chunks_b = chunk_document(st.session_state.pages, strat_b, size_b, overlap_b, threshold_percentile=thresh_b)
            
            st.session_state.compare_results = {
                "chunks_a": chunks_a,
                "chunks_b": chunks_b,
                "params_a": {"strat": strat_a, "size": size_a, "overlap": overlap_a, "thresh": thresh_a},
                "params_b": {"strat": strat_b, "size": size_b, "overlap": overlap_b, "thresh": thresh_b}
            }
            st.success("Comparison calculated successfully!")
            
    if "compare_results" in st.session_state:
        res = st.session_state.compare_results
        chunks_a, chunks_b = res["chunks_a"], res["chunks_b"]
        pa, pb = res["params_a"], res["params_b"]
        
        st.markdown("---")
        st.subheader("📈 Comparative Metrics")
        
        lens_a = [c["char_count"] for c in chunks_a]
        lens_b = [c["char_count"] for c in chunks_b]
        
        avg_a = sum(lens_a) / len(lens_a) if lens_a else 0
        avg_b = sum(lens_b) / len(lens_b) if lens_b else 0
        
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            render_comparison_metric_card(
                f"Config A ({pa['strat']})", len(chunks_a), avg_a, 
                min(lens_a) if lens_a else 0, max(lens_a) if lens_a else 0, 
                "#4f46e5", "#4f46e5"
            )
        with mcol2:
            render_comparison_metric_card(
                f"Config B ({pb['strat']})", len(chunks_b), avg_b, 
                min(lens_b) if lens_b else 0, max(lens_b) if lens_b else 0, 
                "#10b981", "#10b981"
            )
            
        # Distribution Comparison
        st.markdown("### 📊 Distribution Comparison")
        df_a = pd.DataFrame(chunks_a).assign(Config='Config A')
        df_b = pd.DataFrame(chunks_b).assign(Config='Config B')
        combined_df = pd.concat([df_a, df_b])
        
        if not combined_df.empty:
            chart = alt.Chart(combined_df).mark_bar(opacity=0.6, binSpacing=1).encode(
                alt.X("char_count:Q", bin=alt.Bin(maxbins=20), title="Chunk Character Length"),
                y=alt.Y('count()', stack=None, title='Number of Chunks'),
                color=alt.Color('Config:N', scale=alt.Scale(domain=['Config A', 'Config B'], range=['#4f46e5', '#10b981'])),
                tooltip=['Config:N', 'count()']
            ).properties(
                title='Chunk Size Distribution Comparison (Overlay)',
                height=280
            ).configure_title(fontSize=14, anchor='start')
            st.altair_chart(chart, use_container_width=True)
            
        # Visual Boundary Alignment Inspector
        st.markdown("---")
        st.subheader("🔍 Visual Chunk Alignment Inspector")
        st.write("This tool visualizes exactly where the splits occur on the text of page 1.")
        
        page_1_text = st.session_state.pages[0]["text"]
        sample_text = page_1_text[:1500] + "..." if len(page_1_text) > 1500 else page_1_text
        
        spl_a = _split_sample(sample_text, pa)
        spl_b = _split_sample(sample_text, pb)
        
        vis_col1, vis_col2 = st.columns(2)
        with vis_col1:
            st.markdown("#### 🔵 Boundaries (Config A)")
            html_a = render_boundary_highlights(spl_a, "79, 70, 229", "#4f46e5", "A")
            st.markdown(f'<div style="max-height: 400px; overflow-y: auto; border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px;">{html_a}</div>', unsafe_allow_html=True)
            
        with vis_col2:
            st.markdown("#### 🟢 Boundaries (Config B)")
            html_b = render_boundary_highlights(spl_b, "16, 185, 129", "#10b981", "B")
            st.markdown(f'<div style="max-height: 400px; overflow-y: auto; border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px;">{html_b}</div>', unsafe_allow_html=True)
