import os
import streamlit as st
from main import (
    load_documents,
    split_documents,
    create_vector_store,
    load_vector_store,
    create_qa_chain,
    ask_question
)
from config import (
    DOCUMENTS_DIR,
    VECTOR_DB_DIR,
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    LLM_PROVIDER
)

st.set_page_config(page_title="🌾 Agri Crop Management Q&A", page_icon="🌾", layout="wide")

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b5e20;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4a5568;
        margin-bottom: 24px;
    }
    .source-box {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("🌾 Agri RAG Settings")

# Provider selection
provider_option = st.sidebar.radio(
    "Choose LLM Provider:",
    ["Google Gemini (Free Tier)", "OpenAI"],
    index=0 if LLM_PROVIDER == "gemini" else 1
)

active_provider = "gemini" if "Gemini" in provider_option else "openai"
os.environ["LLM_PROVIDER"] = active_provider

if active_provider == "gemini":
    default_key = GEMINI_API_KEY if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here" else ""
    api_key_input = st.sidebar.text_input(
        "Google Gemini API Key:",
        value=default_key,
        type="password",
        help="Get your free API key at https://aistudio.google.com/app/apikey"
    )
    st.sidebar.markdown("🔗 [Get Free Gemini API Key](https://aistudio.google.com/app/apikey)")
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input
        os.environ["GOOGLE_API_KEY"] = api_key_input
else:
    default_key = OPENAI_API_KEY if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here" else ""
    api_key_input = st.sidebar.text_input(
        "OpenAI API Key:",
        value=default_key,
        type="password",
        help="Get your OpenAI key from https://platform.openai.com/api-keys"
    )
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input

# Show documents found in documents/
st.sidebar.markdown("### 📚 Knowledge Base Documents")
if os.path.exists(DOCUMENTS_DIR):
    pdf_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.lower().endswith('.pdf')]
    if pdf_files:
        for f in pdf_files:
            st.sidebar.markdown(f"- 📄 `{f}`")
    else:
        st.sidebar.info("No PDF files found in 'documents/' folder.")
else:
    st.sidebar.warning(f"'{DOCUMENTS_DIR}' directory does not exist.")

st.sidebar.markdown("---")

# Button to build or rebuild the database
if st.sidebar.button("🔄 Build / Rebuild Database", type="primary"):
    current_key = os.getenv("GEMINI_API_KEY") if active_provider == "gemini" else os.getenv("OPENAI_API_KEY")
    if not current_key:
        st.sidebar.error(f"Please provide your {active_provider.title()} API Key above first!")
    else:
        with st.spinner(f"Indexing documents using {active_provider.title()} embeddings..."):
            docs = load_documents(DOCUMENTS_DIR)
            if not docs:
                st.sidebar.error("No PDF documents found in 'documents/' folder.")
            else:
                chunks = split_documents(docs)
                try:
                    vectorstore = create_vector_store(chunks)
                    st.session_state.qa_chain = create_qa_chain(vectorstore)
                    st.sidebar.success("✅ Database successfully built and loaded!")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")

# Main Page Header
st.markdown("<div class='main-header'>🌾 Agri Crop Management Q&A System</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-header'>Ask questions about crop diseases, pest management, and best practices (Powered by <strong>{active_provider.title()}</strong>).</div>", unsafe_allow_html=True)

# Initialize QA chain in session_state if not already present
if 'qa_chain' not in st.session_state:
    try:
        vectorstore = load_vector_store()
        if vectorstore:
            st.session_state.qa_chain = create_qa_chain(vectorstore)
            st.success("✅ Existing knowledge database loaded and ready!")
        else:
            st.session_state.qa_chain = None
            st.info("💡 To begin: Paste your **Gemini API Key** in the left sidebar and click **'Build / Rebuild Database'**.")
    except Exception as e:
        st.session_state.qa_chain = None
        st.info("💡 To begin: Paste your **Gemini API Key** in the left sidebar and click **'Build / Rebuild Database'**.")

# Quick sample questions
st.markdown("**Sample Questions:**")
col1, col2, col3 = st.columns(3)

default_q = ""
if col1.button("🐛 Cotton Pests & Traps"):
    default_q = "What are common cotton pests and how can I manage them?"
if col2.button("🍅 Tomato Blight Prevention"):
    default_q = "How do I prevent fungal diseases and blight in tomatoes?"
if col3.button("🌾 Wheat Irrigation Stages"):
    default_q = "What is the optimal irrigation schedule and critical stages for wheat?"

# Question input field
question = st.text_input("Ask your agricultural question:", value=default_q, placeholder="e.g., How do I manage aphids and whiteflies?")

if st.button("🔍 Get Answer", type="primary") and question:
    if 'qa_chain' in st.session_state and st.session_state.qa_chain is not None:
        with st.spinner("Searching documents & generating answer..."):
            result = ask_question(st.session_state.qa_chain, question)
            
            st.markdown("### 🤖 Answer")
            st.write(result["answer"])
            
            if result["sources"]:
                st.markdown("### 📚 Source References")
                for i, source in enumerate(result["sources"][:3], 1):
                    page = source.metadata.get('page', 'N/A')
                    src_file = os.path.basename(source.metadata.get('source', 'Unknown'))
                    st.markdown(f"<div class='source-box'><strong>{i}. {src_file}</strong> (Page {page})</div>", unsafe_allow_html=True)
    else:
        st.error("Vector database is not initialized yet. Please enter your API key in the left sidebar and click 'Build / Rebuild Database'.")
