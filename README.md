# Medical QA RAG System using LangChain

## Features
-> PDF Document Parsing
Automatically extracts and preprocesses content from PDFs for semantic understanding.

-> Semantic Search with FAISS
Uses vector similarity search to retrieve the most relevant document chunks based on your query.

-> Retrieval-Augmented Generation (RAG)
Combines context-aware retrieval with a powerful LLM to generate accurate, grounded answers.

-> Natural Language Question Answering
Ask complex questions and get human-like, coherent answers tailored to your document.

-> Environment-Variable Secured API Keys
Uses .env file to safely manage access credentials for cloud-based LLMs.

-> Lightweight and Fast
Efficient embedding model (MiniLM-L6-v2) and FAISS make local retrieval quick and scalable.

-> Modular Codebase
Clear separation between ingestion, vector storage, and querying logic—easy to extend or plug in another LLM.

-> Fully Local Embedding & Storage
Embedding and vector storage are done locally, ensuring faster retrieval and data control.

##  Tools Used
- LangChain
- FAISS CPU
- HuggingFace Embeddings and LLM
- Streamlit for UI
- PyMuPDF for PDF parsing
- langchain-google-genai
- uv package

## AI Assistance
- Used Chatgpt for better understanding of architecture of the project

##  Chunking Strategy
- RecursiveCharacterTextSplitter
- chunk size: 500, overlap: 50

##  Workflow
1. `ingest.py`: Loads PDF, chunks it, builds and saves vector DB
2. `query.py`: Loads vector store and LLM to answer any question
3. `app.py`: Streamlit frontend to input question and view answer

##  Instructions
```
uv init
uv venv
uv pip install -r requirements.txt
python ingest.py
python query.py
streamlit run app.py
```

Put your PDF inside `data/sample_pdf.pdf`.

##  Sample Q&A
**Question:** What is the classification code for recurrent depressive disorder in remission?

**Answer:** ICD-10 Code: F33.4 – Recurrent depressive disorder, currently in remission
