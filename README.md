# ✦ SIVA AI — Intelligent Document Assistant

<div align="center">

### 🤖 Chat with Your Documents Using Retrieval-Augmented Generation

An AI-powered document assistant that transforms your PDFs and text files into an interactive knowledge base.

<p>
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Google-Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=for-the-badge" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Database-5B4BDB?style=for-the-badge" />
</p>

**Developed by [Siva Narayana Gali](https://github.com/Gali-3)**

[View Repository](https://github.com/Gali-3/rag-chatbot) • [Connect on LinkedIn](https://www.linkedin.com/in/siva-narayana-gali)

</div>

---

## 🚀 Project Overview

**Siva AI** is an intelligent document question-answering application built using Retrieval-Augmented Generation (RAG).

It allows users to upload PDF and TXT documents, ask questions about their content, and receive context-aware answers generated using Google Gemini.

Instead of manually searching through lengthy documents, users can interact with their documents through a conversational AI interface.

## ✨ Key Features

- 📄 **Document Upload:** Upload PDF and TXT files.
- 🔍 **Document Processing:** Extract and process document content.
- ✂️ **Text Chunking:** Divide documents into smaller meaningful chunks.
- 🧠 **Vector Embeddings:** Convert text into numerical representations.
- 🗄️ **Vector Database:** Store and manage embeddings using ChromaDB.
- 🔎 **Semantic Search:** Retrieve relevant document information.
- 🤖 **AI Answer Generation:** Generate contextual responses using Google Gemini.
- 💬 **Interactive Chat:** Ask questions through a Streamlit interface.
- 📚 **Document Management:** Work with uploaded documents and relevant sources.

## 🏗️ System Architecture

Siva AI follows a Retrieval-Augmented Generation workflow.

### RAG Pipeline

```text
       USER
         |
         v
   Upload Documents
     (PDF / TXT)
         |
         v
    Document Loading
         |
         v
     Text Splitting
         |
         v
  Generate Embeddings
         |
         v
      ChromaDB
    Vector Storage
         |
         |
  USER ASKS QUESTION
         |
         v
   Semantic Retrieval
         |
         v
  Relevant Text Chunks
         |
         v
   Google Gemini AI
         |
         v
   Contextual Answer
         |
         v
    STREAMLIT UI
```

### How It Works

1. **Document Upload:** Users upload PDF or TXT documents.
2. **Text Extraction:** The application extracts document content.
3. **Chunking:** Extracted text is divided into smaller chunks.
4. **Embedding Generation:** Text chunks are converted into vector embeddings.
5. **Vector Storage:** Embeddings are stored in ChromaDB.
6. **Semantic Retrieval:** Relevant chunks are retrieved based on the user's question.
7. **AI Generation:** Retrieved context is passed to Google Gemini.
8. **Response Display:** The generated answer is displayed in the chatbot.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | User interface |
| LangChain | RAG pipeline integration |
| Google Gemini | AI response generation |
| Gemini Embeddings | Vector embedding generation |
| ChromaDB | Vector database |
| PyPDF | PDF document processing |

## 📂 Project Structure

```text
rag-chatbot/
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

*Note: This is the documented logical structure. Adjust filenames and folders to match the actual repository. Environment files, virtual environments and generated database files should not be committed.*

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Gali-3/rag-chatbot.git
```

### 2. Navigate to Project Directory

```bash
cd rag-chatbot
```

### 3. Create Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate Virtual Environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure API Key

Create a `.env` file in the project root directory.

```env
GEMINI_API_KEY=your_gemini_api_key
```

Replace the placeholder with your own API key. Never publish your actual API key on GitHub.

### 7. Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal.

## 💡 Example Use Case

**Scenario:** A user uploads their resume to Siva AI.

**Question:**

"What are the technical skills mentioned in my resume?"

**Expected Application Behavior:**

1. Retrieves relevant sections from the uploaded resume.
2. Passes the retrieved information to Google Gemini.
3. Generates a contextual answer based on the document.

## 🎯 Skills Demonstrated

This project demonstrates practical knowledge of:

- Python programming
- Generative AI
- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Prompt engineering
- Document processing
- Text embeddings
- Semantic search
- Vector databases
- Streamlit application development
- API integration

## 🔮 Future Enhancements

- Support for additional document formats.
- Improved conversational memory.
- Advanced document retrieval techniques.
- User authentication.
- Enhanced response evaluation.
- Multi-document conversational search.

## 👨‍💻 Developer

**Siva Narayana Gali**

Software Developer | Java | Python | AI/ML Enthusiast

- GitHub: [Gali-3](https://github.com/Gali-3)
- LinkedIn: [Siva Narayana Gali](https://www.linkedin.com/in/siva-narayana-gali)

---

<div align="center">

### ✦ SIVA AI

**Turning Documents into Knowledge with Generative AI.**

⭐ If you find this project interesting, consider giving it a star!

*Built with Python, Streamlit, LangChain, Google Gemini and ChromaDB.*

</div>
