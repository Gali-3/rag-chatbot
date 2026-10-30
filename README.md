# ✦ Siva AI – Intelligent Document Assistant

### AI-Powered Document Question Answering System Using Retrieval-Augmented Generation (RAG)

Siva AI is an intelligent document assistant that allows users to upload PDF and TXT documents and ask questions about their content.

The application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from uploaded documents and generate context-based responses using Google Gemini.

---

## 🚀 Project Overview

Searching for specific information in lengthy documents can be time-consuming.

Siva AI simplifies this process by combining document processing, semantic search, vector embeddings, and Large Language Models (LLMs) into an interactive chatbot application.

Users can upload documents, ask questions, and receive answers based on the information retrieved from their documents.

## ✨ Key Features

- 📄 Upload PDF and TXT documents.
- 🔍 Extract and process document content.
- ✂️ Split documents into smaller text chunks.
- 🧠 Generate vector embeddings.
- 🗄️ Store embeddings using ChromaDB.
- 🔎 Retrieve relevant information using semantic search.
- 🤖 Generate context-based answers using Google Gemini.
- 💬 Interactive chatbot interface using Streamlit.
- 📚 Manage uploaded documents and view answer sources.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Backend programming |
| Streamlit | Interactive user interface |
| LangChain | LLM and RAG integration |
| Google Gemini API | AI response generation |
| Gemini Embeddings | Text embedding generation |
| ChromaDB | Vector database |
| PyPDF | PDF text extraction |

## 🏗️ RAG Architecture

The application follows this workflow:

1. **Document Upload:** User uploads a PDF or TXT file.
2. **Document Loading:** Extract text from the uploaded document.
3. **Text Splitting:** Divide the extracted content into smaller chunks.
4. **Embedding Generation:** Convert text chunks into vector embeddings.
5. **Vector Storage:** Store embeddings in ChromaDB.
6. **Question Retrieval:** Search for relevant document chunks using the user's question.
7. **Answer Generation:** Send the retrieved context to Google Gemini.
8. **Final Response:** Display the generated answer in the Streamlit interface.

## 📂 Project Structure

```text
rag-project/
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
│
├── data/
│   ├── java.txt
│   ├── python.txt
│   ├── sql.txt
│   └── machine_learning.txt
│
├── src/
│   ├── load_documents.py
│   ├── split_documents.py
│   ├── create_embeddings.py
│   ├── retrieve_documents.py
│   └── generate_answer.py
│
└── chroma_db/
```

*Note: `.env`, `.venv`, and `chroma_db` are excluded from GitHub where appropriate. The vector database can be created locally when documents are processed.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Gali-3/rag-chatbot.git
```

### 2. Navigate to the project folder

```bash
cd rag-chatbot
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure your Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Replace the placeholder with your own API key. Never upload your actual API key to GitHub.

### 7. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 💡 Example Use Case

**User Question:**

"What are the technical skills mentioned in my resume?"

**Siva AI:**

Retrieves relevant information from the uploaded resume and generates an answer using the retrieved document context.

## 🎯 Learning Outcomes

Through this project, I gained practical experience in:

- Retrieval-Augmented Generation (RAG).
- Document processing and text chunking.
- Embeddings and semantic search.
- Vector database integration.
- Large Language Model integration.
- Building AI applications using Python and Streamlit.

## 🔮 Future Improvements

- Support for additional document formats.
- Improved conversational memory.
- More advanced document retrieval.
- User authentication and document management.
- Enhanced response evaluation.

## 👨‍💻 Developer

**Siva Narayana**

GitHub: https://github.com/Gali-3

---

⭐ If you find this project interesting, feel free to explore the repository and share your feedback.

**Built with Python, Streamlit, LangChain, Google Gemini, and ChromaDB.**