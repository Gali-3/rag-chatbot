
from pathlib import Path
from langchain_community.document_loaders import TextLoader

# Find the data folder
DATA_PATH = Path(__file__).resolve().parent.parent / "data"

# Store loaded documents
documents = []

# Read all text files
for file_path in DATA_PATH.glob("*.txt"):

    loader = TextLoader(
        str(file_path),
        encoding="utf-8"
    )

    loaded_docs = loader.load()
    documents.extend(loaded_docs)

# Display report
print("===== DOCUMENT LOADING REPORT =====")
print("Total documents loaded:", len(documents))

for document in documents:
    print("\nFile:", Path(document.metadata["source"]).name)
    print("Content preview:", document.page_content[:150])
