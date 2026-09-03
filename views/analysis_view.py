import streamlit as st
import pandas as pd
import altair as alt
import json
import time
from chunker import extract_pages_from_pdf, chunk_document
from rag_engine import RAGEngine
from ui_components import render_metrics

def render_analysis_tab(config: dict):
    """Renders Tab 1: Document Upload & Chunking Analysis."""
    st.subheader("Document Upload & Ingestion")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file:
        file_size_kb = f"{uploaded_file.size / 1024:.2f} KB"
        st.write(f"📁 **File:** `{uploaded_file.name}` ({file_size_kb})")
        
        current_params = {
            "name": uploaded_file.name,
            "size": uploaded_file.size,
            "strategy": config["strategy"],
            "chunk_size": config["chunk_size"],
            "chunk_overlap": config["chunk_overlap"],
            "threshold_percentile": config["threshold_percentile"],
            "provider": config["provider"],
            "model": config["ollama_model"] if config["provider"] == "Ollama" else "openai"
        }
        
        # Verify if processing is needed
        needs_processing = (
            st.session_state.chunks is None or 
            st.session_state.last_processed_params != current_params
        )
        
        if needs_processing:
            st.warning("⚠️ Parameters or input file have changed. Click 'Process & Index Document' to rebuild the RAG index.")
            
        if st.button("🚀 Process & Index Document", type="primary"):
            with st.spinner("Extracting text and chunking document..."):
                try:
                    # Extract pages
                    pages = extract_pages_from_pdf(uploaded_file)
                    st.session_state.pages = pages
                    
                    # Generate chunks
                    chunks = chunk_document(
                        pages, 
                        config["strategy"], 
                        config["chunk_size"], 
                        config["chunk_overlap"], 
                        threshold_percentile=config["threshold_percentile"]
                    )
                    st.session_state.chunks = chunks
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.last_processed_params = current_params
                    
                    # Initialize RAG Engine and index
                    st.success("Text chunked successfully. Indexing chunks...")
                    engine = RAGEngine(
                        provider=config["provider"],
                        api_key=config["api_key"],
                        ollama_url=config["ollama_url"],
                        ollama_model=config["ollama_model"]
                    )
                    
                    engine.index_chunks(chunks)
                    st.session_state.rag_engine = engine
                    st.session_state.messages = []
                    
                    st.success("✅ Document processed, chunked, and indexed into vector store!")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                    
        # If chunks exist, show statistics and interactive explorer
        if st.session_state.chunks:
            st.markdown("---")
            st.subheader("📊 Chunking Analytics")
            
            total_pages = len(st.session_state.pages) if st.session_state.pages else 0
            total_chunks = len(st.session_state.chunks)
            char_lens = [c["char_count"] for c in st.session_state.chunks]
            avg_chunk_size = sum(char_lens) / len(char_lens) if char_lens else 0
            
            render_metrics(total_pages, total_chunks, avg_chunk_size)
            
            col_chart, col_download = st.columns([2, 1])
            
            with col_chart:
                df = pd.DataFrame(st.session_state.chunks)
                chart = alt.Chart(df).mark_bar(color='#6366f1').encode(
                    alt.X("char_count:Q", bin=alt.Bin(maxbins=20), title="Chunk Character Length"),
                    y=alt.Y('count()', title='Number of Chunks'),
                    tooltip=['count()']
                ).properties(
                    title='Chunk Size Distribution Histogram',
                    height=280
                ).configure_title(
                    fontSize=14,
                    anchor='start'
                )
                st.altair_chart(chart, use_container_width=True)
                
            with col_download:
                st.markdown("##### 💾 Export Data")
                st.write("Export the generated chunks for inspection or external embedding processes.")
                
                json_data = json.dumps(st.session_state.chunks, indent=2)
                csv_data = df.to_csv(index=False)
                
                st.download_button(
                    label="Download Chunks as JSON",
                    data=json_data,
                    file_name=f"{st.session_state.pdf_name}_chunks.json",
                    mime="application/json"
                )
                st.download_button(
                    label="Download Chunks as CSV",
                    data=csv_data,
                    file_name=f"{st.session_state.pdf_name}_chunks.csv",
                    mime="text/csv"
                )
                
            # Chunk Explorer Table
            st.markdown("---")
            st.subheader("🔍 Chunk Explorer & Search")
            
            search_query = st.text_input("Filter chunks by keyword", "")
            filtered_df = df
            if search_query:
                filtered_df = df[df['text'].str.contains(search_query, case=False, na=False)]
                
            st.write(f"Showing {len(filtered_df)} of {len(df)} chunks")
            st.dataframe(
                filtered_df[['id', 'page_num', 'char_count', 'word_count', 'text']],
                use_container_width=True,
                column_config={
                    "id": st.column_config.NumberColumn("ID", width="small"),
                    "page_num": st.column_config.NumberColumn("Page", width="small"),
                    "char_count": st.column_config.NumberColumn("Chars", width="small"),
                    "word_count": st.column_config.NumberColumn("Words", width="small"),
                    "text": st.column_config.TextColumn("Content", width="large")
                },
                hide_index=True
            )
    else:
        st.info("💡 Please upload a PDF file to begin. The app will extract the text, split it into chunks, and construct an interactive vector search index.")
