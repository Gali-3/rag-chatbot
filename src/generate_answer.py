
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_chroma import Chroma

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)

# Connect to ChromaDB
db_path = Path(__file__).resolve().parent.parent / "chroma_db"

vector_store = Chroma(
    collection_name="my_documents",
    persist_directory=str(db_path),
    embedding_function=embeddings
)

# Initialize Gemini AI
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key
)

# Ask user a question
question = input("Ask your question: ")

# Retrieve relevant document chunks
results = vector_store.similarity_search(question, k=3)

# Combine retrieved content
context = "\n\n".join(
    document.page_content for document in results
)

# Create prompt
prompt = f"""
Answer the question using only the information provided below.

If the answer is not available in the context,
say: "I could not find this information in the documents."

Context:
{context}

Question:
{question}

Answer:
"""

# Generate answer
response = llm.invoke(prompt)

print("\n===== RAG ANSWER =====")
print(response.content)
