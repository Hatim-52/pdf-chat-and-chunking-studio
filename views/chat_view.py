import streamlit as st
import json
from ui_components import render_citation

def render_chat_tab(config: dict):
    """Renders Tab 3: RAG Chat Studio."""
    st.subheader("💬 Chat with your PDF")
    
    if not st.session_state.rag_engine:
        st.info("⚠️ Please upload and process a PDF document in the 'Document & Chunking' tab first to start chatting.")
        return
        
    st.write(f"Currently chatting with: `{st.session_state.pdf_name}` via **{config['provider']}** retriever.")
    
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
