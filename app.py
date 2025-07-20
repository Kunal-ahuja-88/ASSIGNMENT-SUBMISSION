import streamlit as st
from query import answer_query

st.set_page_config(page_title="🧠 Medical RAG QA", layout="centered")
st.title("🩺 Medical Q&A from PDF using LangChain")

query = st.text_input("Ask your medical question:", placeholder="e.g. Classification for depressive disorder")

if st.button("Get Answer") and query:
    with st.spinner("Querying the model..."):
        try:
            result = answer_query(query)
            st.success("✅ Answer:")
            st.markdown(result)
        except Exception as e:
            st.error(f"❌ Error: {e}")
