
import os
import io
import time
import hashlib
import streamlit as st

from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_chroma import Chroma


# ==========================================
# 1. ENVIRONMENT CONFIGURATION
# ==========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


# ==========================================
# 2. STREAMLIT PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Siva AI",
    page_icon="🤖",
    layout="wide"
)


# ==========================================
# 3. SESSION STATE
# ==========================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "upload_notice" not in st.session_state:
    st.session_state.upload_notice = ""

if "delete_notice" not in st.session_state:
    st.session_state.delete_notice = ""


# ==========================================
# 4. APPLICATION HEADING
# ==========================================

st.title("🤖 Siva AI")

st.write(
    "Upload documents, ask questions and get AI-powered answers "
    "from your knowledge base."
)

st.divider()


# ==========================================
# 5. API KEY VALIDATION
# ==========================================

if not api_key:

    st.error(
        "Gemini API key not found. Please check your .env file."
    )

    st.stop()


# ==========================================
# 6. LOAD EMBEDDING MODEL AND VECTOR DATABASE
# ==========================================

@st.cache_resource
def load_vector_database():

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )

    db_path = Path(__file__).resolve().parent / "chroma_db"

    if not db_path.exists():

        raise FileNotFoundError(
            "ChromaDB not found. Run create_embeddings.py first."
        )

    vector_store = Chroma(
        collection_name="my_documents",
        persist_directory=str(db_path),
        embedding_function=embeddings
    )

    return vector_store


# ==========================================
# 7. LOAD GEMINI MODEL
# ==========================================

@st.cache_resource
def load_gemini_model():

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key,
        thinking_level="minimal",
        max_output_tokens=2048,
        timeout=60,
        max_retries=1
    )


# ==========================================
# 8. INITIALIZE RESOURCES
# ==========================================

try:

    vector_store = load_vector_database()

    llm = load_gemini_model()

except Exception as error:

    st.error(f"Application initialization failed: {error}")

    st.stop()


# ==========================================
# 9. HELPER: GET UPLOADED DOCUMENTS
# ==========================================

def get_uploaded_documents():

    data = vector_store.get(
        include=["metadatas"]
    )

    documents = {}

    for metadata in data.get("metadatas", []):

        if not metadata:
            continue

        upload_id = metadata.get("upload_id")

        if upload_id:

            documents[upload_id] = metadata.get(
                "source",
                "Unknown document"
            )

    return documents


# ==========================================
# 10. SIDEBAR - DOCUMENT MANAGEMENT
# ==========================================

st.sidebar.title("⚙️ Settings")

st.sidebar.subheader("📁 Document Library")

try:

    uploaded_documents = get_uploaded_documents()

    st.sidebar.metric(
        "Uploaded documents",
        len(uploaded_documents)
    )

    if uploaded_documents:

        for upload_id, filename in uploaded_documents.items():

            st.sidebar.write(
                f"📄 {filename}"
            )

        st.sidebar.divider()

        st.sidebar.subheader("Delete a document")

        selected_id = st.sidebar.selectbox(
            "Select document",
            options=list(uploaded_documents.keys()),
            format_func=lambda uid: (
                f"{uploaded_documents[uid]} ({uid[:8]})"
            ),
            key="delete_document_select"
        )

        if st.sidebar.button(
            "🗑️ Delete Selected Document",
            use_container_width=True
        ):

            vector_store.delete(
                where={"upload_id": selected_id}
            )

            st.session_state.delete_notice = (
                "Selected document deleted successfully."
            )

            st.rerun()

    else:

        st.sidebar.info("No uploaded documents yet.")

except Exception as error:

    st.sidebar.error(
        f"Could not load document library: {error}"
    )


# ==========================================
# 11. SIDEBAR - CHAT MANAGEMENT
# ==========================================

st.sidebar.divider()

st.sidebar.subheader("💬 Chat Management")

if st.sidebar.button(
    "Clear Chat History",
    use_container_width=True
):

    st.session_state.chat_history = []

    st.rerun()


st.sidebar.divider()

st.sidebar.caption(
    "Built with Python, Streamlit, LangChain, "
    "Gemini AI and ChromaDB."
)


# ==========================================
# 12. DISPLAY STATUS MESSAGES
# ==========================================

if st.session_state.upload_notice:

    st.success(st.session_state.upload_notice)

    st.session_state.upload_notice = ""

if st.session_state.delete_notice:

    st.success(st.session_state.delete_notice)

    st.session_state.delete_notice = ""


# ==========================================
# 13. MULTIPLE DOCUMENT UPLOAD
# ==========================================

st.subheader("📂 Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDF or TXT documents",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:

    st.info(
        f"{len(uploaded_files)} document(s) selected."
    )

    if st.button(
        "🚀 Process All Documents",
        type="primary"
    ):

        processed_count = 0
        skipped_count = 0
        failed_count = 0

        progress_bar = st.progress(0)

        for index, uploaded_file in enumerate(uploaded_files):

            try:

                with st.spinner(
                    f"Processing {uploaded_file.name}..."
                ):

                    file_bytes = uploaded_file.getvalue()

                    # Generate unique file ID
                    file_hash = hashlib.sha256(
                        file_bytes
                    ).hexdigest()

                    # Check for duplicates in ChromaDB
                    existing = vector_store.get(
                        where={"upload_id": file_hash},
                        include=["metadatas"]
                    )

                    if existing.get("ids"):

                        st.info(
                            f"{uploaded_file.name} is already processed."
                        )

                        skipped_count += 1

                        progress_bar.progress(
                            (index + 1) / len(uploaded_files)
                        )

                        continue

                    # --------------------------
                    # Extract PDF text by page
                    # --------------------------

                    page_documents = []

                    if uploaded_file.name.lower().endswith(".pdf"):

                        pdf_reader = PdfReader(
                            io.BytesIO(file_bytes)
                        )

                        for page_number, page in enumerate(
                            pdf_reader.pages,
                            start=1
                        ):

                            page_text = page.extract_text() or ""

                            if page_text.strip():

                                page_documents.append(
                                    Document(
                                        page_content=page_text,
                                        metadata={
                                            "source": uploaded_file.name,
                                            "upload_id": file_hash,
                                            "page": page_number
                                        }
                                    )
                                )

                    # --------------------------
                    # Extract TXT text
                    # --------------------------

                    else:

                        text_content = file_bytes.decode(
                            "utf-8-sig",
                            errors="replace"
                        )

                        if text_content.strip():

                            page_documents.append(
                                Document(
                                    page_content=text_content,
                                    metadata={
                                        "source": uploaded_file.name,
                                        "upload_id": file_hash,
                                        "page": 1
                                    }
                                )
                            )

                    # Check extracted content
                    if not page_documents:

                        st.warning(
                            f"No readable text found in "
                            f"{uploaded_file.name}."
                        )

                        failed_count += 1

                        progress_bar.progress(
                            (index + 1) / len(uploaded_files)
                        )

                        continue

                    # --------------------------
                    # Split documents into chunks
                    # --------------------------

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=700,
                        chunk_overlap=100
                    )

                    chunks = splitter.split_documents(
                        page_documents
                    )

                    # --------------------------
                    # Generate unique chunk IDs
                    # --------------------------

                    chunk_ids = [
                        f"{file_hash}_{chunk_index}"
                        for chunk_index in range(len(chunks))
                    ]

                    # --------------------------
                    # Store in ChromaDB
                    # --------------------------

                    vector_store.add_documents(
                        documents=chunks,
                        ids=chunk_ids
                    )

                    processed_count += 1

                    st.success(
                        f"{uploaded_file.name} processed successfully. "
                        f"{len(chunks)} chunks created."
                    )

            except Exception as error:

                failed_count += 1

                st.error(
                    f"Error processing {uploaded_file.name}: {error}"
                )

            progress_bar.progress(
                (index + 1) / len(uploaded_files)
            )

        st.session_state.upload_notice = (
            f"Processing completed. "
            f"Processed: {processed_count}, "
            f"Already processed: {skipped_count}, "
            f"Failed: {failed_count}."
        )

        st.rerun()


st.divider()


# ==========================================
# 14. DISPLAY CHAT HISTORY
# ==========================================

st.subheader("💬 Conversation History")

if st.session_state.chat_history:

    for chat in st.session_state.chat_history:

        with st.chat_message("user"):

            st.write(chat["question"])

        with st.chat_message("assistant"):

            st.markdown(chat["answer"])

            # Performance details
            with st.expander("⏱️ Performance Report"):

                st.write(
                    f"Retrieval: {chat['retrieval_time']:.2f} seconds"
                )

                st.write(
                    f"Gemini generation: "
                    f"{chat['generation_time']:.2f} seconds"
                )

                st.write(
                    f"Total: {chat['total_time']:.2f} seconds"
                )

            # Source details
            with st.expander("📚 View Sources"):

                if chat["sources"]:

                    for source in chat["sources"]:

                        st.write(
                            f"📄 {source['name']} | "
                            f"Page: {source['page']}"
                        )

                        st.write(source["content"])

                        st.divider()

                else:

                    st.write("No source documents retrieved.")

else:

    st.info(
        "No conversations yet. Ask your first question below."
    )


# ==========================================
# 15. USER QUESTION INPUT
# ==========================================

st.divider()

st.subheader("🤖 Ask Your Documents")

question = st.chat_input(
    "Ask something about your uploaded documents..."
)


# ==========================================
# 16. RETRIEVE DOCUMENTS AND GENERATE ANSWER
# ==========================================

if question:

    try:

        # --------------------------
        # Document retrieval
        # --------------------------

        with st.spinner("Searching your documents..."):

            retrieval_start = time.perf_counter()

            results = vector_store.similarity_search(
                question,
                k=5
            )

            retrieval_time = (
                time.perf_counter() - retrieval_start
            )

        # Prepare source details
        sources = []

        context_parts = []

        for doc in results:

            source_name = doc.metadata.get(
                "source",
                "Unknown"
            )

            page_number = doc.metadata.get(
                "page",
                "Unknown"
            )

            context_parts.append(
                f"""
Source: {source_name}
Page: {page_number}

Content:
{doc.page_content}
"""
            )

            sources.append({
                "name": source_name,
                "page": page_number,
                "content": doc.page_content
            })

        context = "\n\n".join(context_parts)

        # --------------------------
        # Generate Gemini answer
        # --------------------------

        if not results:

            answer = (
                "I could not find relevant information "
                "in the available documents."
            )

            generation_time = 0

        else:

            prompt = f"""
You are an intelligent AI document assistant.

Your task is to answer the user's question using
ONLY the retrieved document context below.

Instructions:

1. Give clear and beginner-friendly answers.
2. Use bullet points when appropriate.
3. Do not invent information.
4. Do not assume facts that are not in the documents.
5. If the answer is missing, clearly say:
   "I could not find this information in the documents."
6. Mention the source document and page when relevant.

Retrieved document context:

{context}

User question:

{question}

Answer:
"""

            with st.spinner("Gemini is generating your answer..."):

                generation_start = time.perf_counter()

                response = llm.invoke(prompt)

                generation_time = (
                    time.perf_counter() - generation_start
                )

            answer = response.content

            # Normalize response
            if isinstance(answer, list):

                answer = "\n".join(
                    item.get("text", "")
                    for item in answer
                    if isinstance(item, dict)
                    and item.get("type") == "text"
                )

            if not isinstance(answer, str):

                answer = str(answer)

        # --------------------------
        # Calculate total time
        # --------------------------

        total_time = (
            retrieval_time + generation_time
        )

        # --------------------------
        # Save conversation
        # --------------------------

        st.session_state.chat_history.append({

            "question": question,

            "answer": answer,

            "retrieval_time": retrieval_time,

            "generation_time": generation_time,

            "total_time": total_time,

            "sources": sources

        })

        # Refresh to display new conversation
        st.rerun()

    except Exception as error:

        st.error(
            f"Question processing failed: {error}"
        )


# ==========================================
# 17. FOOTER
# ==========================================

st.divider()

st.caption(
    "AI Document Assistant | Python | Streamlit | "
    "LangChain | Gemini AI | ChromaDB"
)
