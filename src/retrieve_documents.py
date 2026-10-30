
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Create embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)

# Connect to existing ChromaDB
db_path = Path(__file__).resolve().parent.parent / "chroma_db"

vector_store = Chroma(
    collection_name="my_documents",
    persist_directory=str(db_path),
    embedding_function=embeddings
)

# Ask a question
question = "What is inheritance in Java?"

# Search for the 3 most relevant chunks
results = vector_store.similarity_search(
    question,
    k=3
)

print("\n===== RETRIEVAL REPORT =====")
print("Question:", question)

for index, document in enumerate(results):
    print(f"\nResult {index + 1}")
    print("Source:", Path(document.metadata["source"]).name)
    print("Content:", document.page_content)
