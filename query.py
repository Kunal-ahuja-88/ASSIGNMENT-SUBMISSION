from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
apikey=os.getenv("GEMINI_API_KEY")
print(apikey)

def load_vectorstore(path="vectorstore"):
    """
    Load the vector store from the local file system.
    Uses HuggingFace sentence-transformer for embedding.
    """
     
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

def answer_query(query_text):
    """
    Given a user query, retrieve relevant documents from the FAISS vector store,
    and use the LLM to generate an answer based on retrieved context.
    """

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = load_llm()

    # Create a RetrievalQA chain combining LLM and retriever
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
    return qa.run(query_text) # Run the QA chain

if __name__ == "__main__":
    query = "Give me the correct coded classification for the following diagnosis: Recurrent depressive disorder, currently in remission"
    print("Answer:", answer_query(query))
