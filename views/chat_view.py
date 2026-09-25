import streamlit as st
import json
from ui.ui_components import render_citation

def render_chat_tab(config: dict):
    """Renders Tab 3: RAG Chat Studio."""
    st.subheader("💬 RAG Chat Studio")
    
    if not st.session_state.rag_engine:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%);
                    border: 1px dashed rgba(255, 255, 255, 0.15); border-radius: 20px; padding: 2.8rem 2rem;
                    text-align: center; margin-top: 1.5rem; backdrop-filter: blur(16px); box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);">
            <div style="font-size: 2.8rem; margin-bottom: 0.8rem; filter: drop-shadow(0 0 15px rgba(56, 189, 248, 0.4));">💬</div>
            <div style="font-family: 'Outfit', sans-serif; font-size: 1.35rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.5rem;">
                Neural Retrieval Engine Inactive
            </div>
            <div style="color: #94a3b8; font-size: 0.95rem; max-width: 580px; margin: 0 auto 1.6rem auto; line-height: 1.6;">
                Please upload and index a PDF document in the <strong>Document &amp; Chunking Analysis</strong> tab first. 
                Once indexed, you can interrogate passages with grounded source citations and similarity scores.
            </div>
            <div style="display: inline-flex; gap: 0.75rem; flex-wrap: wrap; justify-content: center;">
                <span class="hero-pill-badge" style="background: rgba(56, 189, 248, 0.1); border-color: rgba(56, 189, 248, 0.25);">🔍 Vector Similarity Search</span>
                <span class="hero-pill-badge" style="background: rgba(99, 102, 241, 0.1); border-color: rgba(99, 102, 241, 0.25);">📜 In-line Page Citations</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
        
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.75rem;
                background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); 
                border-radius: 14px; padding: 0.75rem 1.2rem; margin-bottom: 1.5rem; backdrop-filter: blur(12px);">
        <div style="display: flex; align-items: center; gap: 0.6rem;">
            <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: #34d399; box-shadow: 0 0 8px #34d399;"></span>
            <span style="color: #e2e8f0; font-size: 0.88rem; font-weight: 600;">Active Knowledge Base:</span>
            <span style="color: #c7d2fe; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; background: rgba(99, 102, 241, 0.15); padding: 0.2rem 0.5rem; border-radius: 6px; border: 1px solid rgba(99, 102, 241, 0.3);">{st.session_state.pdf_name}</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span class="citation-badge" style="background: rgba(168, 85, 247, 0.12); border-color: rgba(168, 85, 247, 0.3); color: #d8b4fe;">Retriever: {config['provider']}</span>
            <span class="citation-badge" style="background: rgba(56, 189, 248, 0.12); border-color: rgba(56, 189, 248, 0.3); color: #7dd3fc;">Top K: {config['top_k']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("🔍 View Retrieved Sources"):
                    for idx, src in enumerate(msg["sources"]):
                        render_citation(src, idx)
                        
    # Accept chat input
    user_query = st.chat_input("Ask a question about the document...")
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant passages..."):
                retrieved_chunks = st.session_state.rag_engine.retrieve(
                    user_query, 
                    top_k=config["top_k"], 
                    similarity_threshold=config["similarity_threshold"]
                )
                
            response_placeholder = st.empty()
            full_response = ""
            try:
                for chunk in st.session_state.rag_engine.generate_response(
                    query=user_query,
                    retrieved_chunks=retrieved_chunks,
                    chat_history=st.session_state.messages[:-1],
                    system_prompt_template=config["system_prompt_template"] if config["system_prompt_template"] else None,
                    openai_model=config["openai_model"]
                ):
                    full_response += chunk
                    response_placeholder.write(full_response + "▌")
                response_placeholder.write(full_response)
            except Exception as e:
                error_msg = f"An error occurred during generation: {str(e)}"
                response_placeholder.write(error_msg)
                full_response = error_msg
            
            if retrieved_chunks:
                with st.expander("🔍 View Retrieved Sources"):
                    for idx, src in enumerate(retrieved_chunks):
                        render_citation(src, idx)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": retrieved_chunks
            })
            
        st.rerun()
        
    # Chat utility options
    if st.session_state.messages:
        st.markdown("---")
        ut_col1, ut_col2 = st.columns(2)
        with ut_col1:
            chat_history_json = json.dumps(st.session_state.messages, indent=2)
            st.download_button(
                label="📥 Download Chat History (JSON)",
                data=chat_history_json,
                file_name=f"{st.session_state.pdf_name}_chat_history.json",
                mime="application/json"
            )
        with ut_col2:
            with st.popover("🗑️ Clear Chat History"):
                st.write("Are you sure you want to clear the entire chat history?")
                if st.button("Yes, Clear Chat", type="primary", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()
