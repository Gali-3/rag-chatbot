
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load Gemini API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found")

# Load documents
data_path = Path(__file__).resolve().parent.parent / "data"
documents = []

for file_path in data_path.glob("*.txt"):
    loader = TextLoader(str(file_path), encoding="utf-8")
    documents.extend(loader.load())

print("Documents loaded:", len(documents))

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Total chunks:", len(chunks))

# Create embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)

# Store vectors in ChromaDB
db_path = Path(__file__).resolve().parent.parent / "chroma_db"

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(db_path),
    collection_name="my_documents"
)

print("\n===== VECTOR DATABASE REPORT =====")
print("Embeddings created and stored successfully!")
print("Database location:", db_path)
