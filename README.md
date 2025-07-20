# Medical QA RAG System using LangChain

## ✅ Features
- LangChain-based RAG pipeline
- FAISS vector store
- Embeddings: all-MiniLM-L6-v2
- LLM: mistralai/Mistral-7B-Instruct-v0.1
- Streamlit interface
- `.uv` compatible setup (no venv needed)

## 🛠️ Tools Used
- LangChain
- FAISS
- HuggingFace Embeddings and LLM
- Streamlit for UI
- PyMuPDF for PDF parsing

## AI Assistance
- Used Chatgpt for better understanding of architecture of the project

## 📄 Chunking Strategy
- RecursiveCharacterTextSplitter
- chunk size: 500, overlap: 50

## 🔄 Workflow
1. `ingest.py`: Loads PDF, chunks it, builds and saves vector DB
2. `query.py`: Loads vector store and LLM to answer any question
3. `app.py`: Streamlit frontend to input question and view answer

## 🚀 Instructions
```
uv init
uv venv
uv pip install -r requirements.txt
python ingest.py
python query.py
streamlit run app.py
```

Put your PDF inside `data/your_pdf.pdf`.

## 🧠 Sample Q&A
**Question:** What is the classification code for recurrent depressive disorder in remission?

**Answer:** ICD-10 Code: F33.4 – Recurrent depressive disorder, currently in remission
