"""
Knat LLM — Project 4: minimal Streamlit chat UI over the RAG pipeline.
"""
import streamlit as st

from pipeline.rag_pipeline import build_rag_pipeline, ask

st.set_page_config(page_title="Knat LLM — RAG Assistant", page_icon="📄")
st.title("📄 Knat LLM — Document RAG Assistant")
st.caption("Ask questions grounded in the PDFs ingested into the Elasticsearch index.")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = build_rag_pipeline()

if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.write(content)

question = st.chat_input("Ask a question about your ingested documents...")
if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating an answer..."):
            result = ask(st.session_state.pipeline, question)
        st.write(result["answer"])
        st.caption(f"Latency: {result['latency_seconds']}s")

    st.session_state.history.append(("assistant", result["answer"]))
