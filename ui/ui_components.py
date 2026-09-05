import streamlit as st
import html
import os

def load_css(file_name="style.css"):
    """Loads a CSS file and injects it into Streamlit."""
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_header():
    """Renders the top application header."""
    st.markdown("""
    <div class="title-container">
        <h1 class="app-title">📄 PDF Chat & Chunking Studio</h1>
        <p class="app-subtitle">Upload documents, optimize chunking strategies, and chat with your data using local or cloud RAG.</p>
    </div>
    """, unsafe_allow_html=True)

def render_metrics(total_pages: int, total_chunks: int, avg_chunk_size: float):
    """Renders core metrics cards for document analysis."""
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card">
            <div class="metric-value">{total_pages}</div>
            <div class="metric-label">Total Pages</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{total_chunks}</div>
            <div class="metric-label">Total Chunks</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{avg_chunk_size:.1f}</div>
            <div class="metric-label">Avg Chunk Size (Chars)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_comparison_metric_card(title: str, total_chunks: int, avg_size: float, min_size: int, max_size: int, border_color: str, value_color: str):
    """Renders a comparative metric card for Strategy Comparison."""
    st.markdown(f"""
    <div class="metric-card" style="border-left: 5px solid {border_color}; margin-bottom: 1rem; text-align: left; padding: 1rem;">
        <div style="font-weight: 700; font-size: 1.1rem; color: {border_color};">{title}</div>
        <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
            <div>
                <div class="metric-value" style="font-size: 1.5rem; text-align: center; color: {value_color};">{total_chunks}</div>
                <div class="metric-label" style="font-size: 0.75rem; text-align: center;">Total Chunks</div>
            </div>
            <div>
                <div class="metric-value" style="font-size: 1.5rem; color: {value_color}; text-align: center;">{avg_size:.1f}</div>
                <div class="metric-label" style="font-size: 0.75rem; text-align: center;">Avg Size (chars)</div>
            </div>
            <div>
                <div class="metric-value" style="font-size: 1.3rem; color: #64748b; text-align: center;">{min_size}/{max_size}</div>
                <div class="metric-label" style="font-size: 0.75rem; text-align: center;">Min/Max Chunks</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_citation(source: dict, index: int):
    """Renders a formatted source citation block."""
    st.markdown(f"""
    <div class="citation-container">
        <div class="citation-header">
            Source {index + 1} - Page {source['page_num']}
            <span class="citation-score">Similarity Score: {source['score']:.4f}</span>
        </div>
        <div>{source['text']}</div>
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
            f'<div style="background-color: {bg_color}; border-left: 4px solid {border_color}; '
            f'padding: 8px; margin: 6px 0; border-radius: 0 6px 6px 0; font-family: monospace; font-size: 0.85rem;">'
            f'<strong style="color: {border_color}; font-size: 0.75rem; display: block; margin-bottom: 2px;">'
            f'{title_prefix} - Chunk {idx+1} ({len(chunk)} chars)</strong>'
            f'{escaped_chunk}</div>'
        )
    return "\n".join(html_elements)
