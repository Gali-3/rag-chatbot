
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Find the data folder
DATA_PATH = Path(__file__).resolve().parent.parent / "data"

# Load all documents
documents = []

for file_path in DATA_PATH.glob("*.txt"):
    loader = TextLoader(str(file_path), encoding="utf-8")
    documents.extend(loader.load())

print("Total documents loaded:", len(documents))

# Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

# Split documents into chunks
chunks = text_splitter.split_documents(documents)

# Display the result
print("\n===== DOCUMENT SPLITTING REPORT =====")
print("Total chunks created:", len(chunks))

for index, chunk in enumerate(chunks):
    print(f"\nChunk {index + 1}")
    print("Source:", Path(chunk.metadata["source"]).name)
    print("Content:", chunk.page_content)
