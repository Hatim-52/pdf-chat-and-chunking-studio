import streamlit as st
import html
import os

def load_css(file_name="style.css"):
    """Loads a CSS file and injects it into Streamlit."""
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_header():
    """Renders the top application hero header inspired by Lumina & Slate aesthetics."""
    st.markdown("""
    <div class="hero-wrapper">
        <div class="hero-badge-container">
            <div class="hero-pill-badge">
                <span class="hero-pulse-dot"></span>
                Pure Insight &bull; Zero Noise
            </div>
            <div class="hero-pill-badge" style="background: rgba(168, 85, 247, 0.12); border-color: rgba(168, 85, 247, 0.3); color: #e9d5ff;">
                Neural Chunking Studio 2.0
            </div>
        </div>
        <h1 class="hero-title">
            Document Intelligence.<br>
            <span class="hero-gradient-text">Engineered for Precision RAG.</span>
        </h1>
        <p class="hero-subtitle">
            Upload complex documents, optimize chunk boundaries across multiple splitting strategies, 
            and interrogate your unstructured knowledge with high-fidelity vector retrieval.
        </p>
        <div class="hero-dock">
            <div class="hero-dock-item">
                <div class="dock-icon">⚡</div>
                <div class="dock-content">
                    <span class="dock-title">Hybrid Retrievers</span>
                    <span class="dock-desc">TF-IDF &bull; Ollama &bull; OpenAI</span>
                </div>
            </div>
            <div class="hero-dock-item">
                <div class="dock-icon">📐</div>
                <div class="dock-content">
                    <span class="dock-title">Adaptive Strategies</span>
                    <span class="dock-desc">Semantic &bull; Recursive &bull; Sentence</span>
                </div>
            </div>
            <div class="hero-dock-item">
                <div class="dock-icon">🔬</div>
                <div class="dock-content">
                    <span class="dock-title">Deep Inspection</span>
                    <span class="dock-desc">Token Histograms &amp; Splits</span>
                </div>
            </div>
            <div class="hero-dock-item">
                <div class="dock-icon">🛡️</div>
                <div class="dock-content">
                    <span class="dock-title">Zero Leakage</span>
                    <span class="dock-desc">100% Offline Local Processing</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_metrics(total_pages: int, total_chunks: int, avg_chunk_size: float):
    """Renders core metrics cards for document analysis with frosted glass styling."""
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card">
            <div class="metric-value">{total_pages:,}</div>
            <div class="metric-label">📄 Extracted Pages</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{total_chunks:,}</div>
            <div class="metric-label">🧩 Indexed Chunks</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{avg_chunk_size:.1f}</div>
            <div class="metric-label">📏 Avg Chunk Size (Chars)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_comparison_metric_card(title: str, total_chunks: int, avg_size: float, min_size: int, max_size: int, border_color: str, value_color: str):
    """Renders a comparative telemetry card for Strategy Comparison."""
    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid {border_color}; text-align: left; padding: 1.25rem 1.4rem; margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div style="font-weight: 700; font-size: 1.1rem; color: #f8fafc; font-family: 'Outfit', sans-serif;">
                {title}
            </div>
            <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: {border_color}; box-shadow: 0 0 10px {border_color};"></span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.75rem; text-align: center;">
            <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 0.6rem;">
                <div class="metric-value" style="font-size: 1.45rem; color: {value_color}; margin: 0;">{total_chunks}</div>
                <div class="metric-label" style="font-size: 0.7rem; margin-top: 0.2rem;">Total Chunks</div>
            </div>
            <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 0.6rem;">
                <div class="metric-value" style="font-size: 1.45rem; color: {value_color}; margin: 0;">{avg_size:.1f}</div>
                <div class="metric-label" style="font-size: 0.7rem; margin-top: 0.2rem;">Avg Size (Ch)</div>
            </div>
            <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 0.6rem;">
                <div class="metric-value" style="font-size: 1.25rem; color: #cbd5e1; margin: 0;">{min_size}/{max_size}</div>
                <div class="metric-label" style="font-size: 0.7rem; margin-top: 0.2rem;">Min / Max</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_citation(source: dict, index: int):
    """Renders a formatted source citation block inspired by Lumina's glass cards."""
    page_num = source.get('page_num', 'N/A')
    score = source.get('score', 0.0)
    escaped_text = html.escape(source.get('text', ''))
    
    st.markdown(f"""
    <div class="citation-container">
        <div class="citation-header">
            <span class="citation-badge">
                <span>📄</span> Page {page_num} &bull; Reference #{index + 1}
            </span>
            <span class="citation-score">
                Similarity {score:.4f}
            </span>
        </div>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.88rem; line-height: 1.6; color: #e2e8f0;">
            {escaped_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_boundary_highlights(splits: list[str], base_color_rgba: str, border_color_hex: str, title_prefix: str) -> str:
    """Generates color-coded HTML blocks showing split chunks for Visual Alignment."""
    html_elements = []
    colors_list = [
        (f"rgba({base_color_rgba}, 0.12)", border_color_hex),
        (f"rgba({base_color_rgba}, 0.05)", border_color_hex)
    ]
    for idx, chunk in enumerate(splits):
        bg_color, border_color = colors_list[idx % len(colors_list)]
        escaped_chunk = html.escape(chunk).replace("\n", "<br>")
        html_elements.append(
            f'<div style="background-color: {bg_color}; border-left: 3px solid {border_color}; '
            f'padding: 10px 12px; margin: 8px 0; border-radius: 0 10px 10px 0; font-family: \'JetBrains Mono\', monospace; font-size: 0.82rem; line-height: 1.5; color: #e2e8f0; border-top: 1px solid rgba(255,255,255,0.03); border-bottom: 1px solid rgba(255,255,255,0.03); border-right: 1px solid rgba(255,255,255,0.03);">'
            f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">'
            f'<strong style="color: {border_color}; font-size: 0.75rem; letter-spacing: 0.03em;">'
            f'{title_prefix} &bull; Chunk #{idx+1}</strong>'
            f'<span style="font-size: 0.72rem; color: #94a3b8; background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 4px;">{len(chunk)} chars</span>'
            f'</div>'
            f'{escaped_chunk}</div>'
        )
    return "\n".join(html_elements)
