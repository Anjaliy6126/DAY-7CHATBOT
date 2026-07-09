"""
Streamlit RAG Chatbot for Samsung Washing Machine Manual
----------------------------------------------------------
Loads a fixed washing-machine manual (HTML file) already stored in the repo,
then lets you ask questions about it.
Uses LangChain + Chroma + OpenAI for Retrieval Augmented Generation.

Run with:
    streamlit run day_7.py
"""

import os

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# ----------------------------------------------------------------------
# Fixed path to the manual already included in the repo.
# Must match the exact filename (spelling, spacing, capitalization, extension).
# ----------------------------------------------------------------------
MANUAL_PATH = "How to use the various modes of the washing machine _ Samsung LEVANT.html"

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Washing Machine Manual Chatbot",
    page_icon="🧺",
    layout="centered",
)

st.title("🧺 Samsung Washing Machine Manual Chatbot")
st.caption(
    "Ask a question about the Samsung washing machine manual and get a "
    "context-aware answer powered by Retrieval Augmented Generation (RAG)."
)

# ----------------------------------------------------------------------
# Sidebar: API key + settings (no file upload needed — manual is fixed)
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Setup")

    # Never hardcode API keys in source. Prefer st.secrets, fall back to
    # a password-masked text input so the key is never persisted or shown.
    api_key = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
    if not api_key:
        api_key = st.text_input("OpenAI API Key", type="password")

    chunk_size = st.slider("Chunk size", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, step=50)
    temperature = st.slider("Model temperature", 0.0, 1.0, 0.0, step=0.1)

    build_clicked = st.button("Build knowledge base", type="primary")

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------------------------------------------------
# Build the RAG pipeline
# ----------------------------------------------------------------------
def build_rag_chain(html_path: str, api_key: str, chunk_size: int, chunk_overlap: int, temperature: float):
    os.environ["OPENAI_API_KEY"] = api_key

    # Load the HTML document
    loader = UnstructuredHTMLLoader(file_path=html_path)
    machine_docs = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(machine_docs)

    # Embeddings + vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature, api_key=api_key)

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:"""
    )

    # Chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return rag_chain


if build_clicked:
    if not api_key:
        st.sidebar.error("Please provide your OpenAI API key.")
    elif not os.path.exists(MANUAL_PATH):
        st.sidebar.error(
            f"Manual file not found at '{MANUAL_PATH}'. "
            "Make sure the HTML file is in the same folder as this script "
            "and the filename matches exactly."
        )
    else:
        with st.spinner("Reading manual, chunking text, and building the vector store..."):
            try:
                st.session_state.rag_chain = build_rag_chain(
                    MANUAL_PATH, api_key, chunk_size, chunk_overlap, temperature
                )
                st.session_state.messages = []
                st.sidebar.success("Knowledge base ready! Ask a question below.")
            except Exception as e:
                st.sidebar.error(f"Error building knowledge base: {e}")

# ----------------------------------------------------------------------
# Chat interface
# ----------------------------------------------------------------------
st.divider()

if st.session_state.rag_chain is None:
    st.info("Click **Build knowledge base** in the sidebar to get started.")
else:
    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    query = st.chat_input("Ask a question about the manual, e.g. 'What is the cycle for DRUM CLEAN?'")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = st.session_state.rag_chain.invoke(query).content
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Error generating answer: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
