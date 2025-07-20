import fitz  # PyMuPDF
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import os

## Extract all text from a PDF file
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Full ingestion pipeline: PDF → Chunks → Embeddings → FAISS Vectorstore
def ingest(path="data/sample_pdf.pdf", faiss_path="vectorstore"):
     # Step 1: Extract text from PDF
    text = extract_text_from_pdf(path)

     # Step 2: Wrap the extracted text into a LangChain Document object
    docs = [Document(page_content=text)]

    # Step 3: Split the document into smaller overlapping chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

     # Step 4: Create embeddings using HuggingFace sentence-transformer model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step 5: Create a FAISS vector store from the embedded document chunks
    db = FAISS.from_documents(split_docs, embeddings)

    # Step 6: Save the vector store to disk for later retrieval
    db.save_local(faiss_path)
    print(f"✅ Ingestion complete and saved to {faiss_path}")

if __name__ == "__main__":
    ingest()
