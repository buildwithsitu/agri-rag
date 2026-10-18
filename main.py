import os
import sys

# Ensure UTF-8 output on Windows terminals to prevent charmap encoding errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    from langchain.document_loaders import PyPDFLoader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embedding and LLM imports for Gemini and OpenAI
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
except ImportError:
    ChatGoogleGenerativeAI = None
    GoogleGenerativeAIEmbeddings = None

try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI
except ImportError:
    OpenAIEmbeddings = None
    ChatOpenAI = None
    OpenAI = None

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

try:
    from langchain_classic.chains import RetrievalQA
except ImportError:
    try:
        from langchain.chains import RetrievalQA
    except ImportError:
        from langchain.chains.retrieval_qa.base import RetrievalQA

from config import (
    DOCUMENTS_DIR,
    VECTOR_DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    NUM_RETRIEVED_DOCS,
    TEMPERATURE,
    LLM_PROVIDER,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_EMBEDDING_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL
)


def get_embeddings():
    """Get embedding function based on LLM_PROVIDER."""
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().lower()
    
    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or GEMINI_API_KEY
        if not key or key == "your_gemini_api_key_here":
            raise ValueError("GEMINI_API_KEY is not configured! Get your free key at https://aistudio.google.com/app/apikey and add it to .env or the UI.")
        if GoogleGenerativeAIEmbeddings is None:
            raise ImportError("langchain-google-genai is not installed.")
        return GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDING_MODEL,
            google_api_key=key
        )
    else:
        key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
        if not key or key == "your_openai_api_key_here":
            raise ValueError("OPENAI_API_KEY is not configured in .env!")
        return OpenAIEmbeddings(openai_api_key=key)


def get_llm():
    """Get LLM instance based on LLM_PROVIDER."""
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().lower()
    
    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or GEMINI_API_KEY
        if not key or key == "your_gemini_api_key_here":
            raise ValueError("GEMINI_API_KEY is not configured! Get your free key at https://aistudio.google.com/app/apikey")
        if ChatGoogleGenerativeAI is None:
            raise ImportError("langchain-google-genai is not installed.")
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=key,
            temperature=TEMPERATURE
        )
    else:
        key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
        if not key or key == "your_openai_api_key_here":
            raise ValueError("OPENAI_API_KEY is not configured in .env!")
        return ChatOpenAI(
            openai_api_key=key,
            temperature=TEMPERATURE,
            model_name=OPENAI_MODEL
        )


def load_documents(directory):
    """Load all PDF documents from the specified directory."""
    documents = []
    
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist!")
        return documents
    
    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in '{directory}'")
        return documents
    
    print(f"Found {len(pdf_files)} PDF file(s)")
    for filename in pdf_files:
        file_path = os.path.join(directory, filename)
        print(f"Loading: {filename}")
        try:
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()
            documents.extend(loaded_docs)
            print(f"  [OK] Loaded {len(loaded_docs)} pages")
        except Exception as e:
            print(f"  [Error] Failed to load {filename}: {str(e)}")
    
    print(f"\nTotal documents loaded: {len(documents)} pages")
    return documents


def split_documents(documents):
    """Split documents into smaller chunks for retrieval."""
    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    print(f"Configured chunk size: {CHUNK_SIZE} characters, overlap: {CHUNK_OVERLAP}")
    return chunks


def create_vector_store(chunks):
    """Create a persistent vector database from document chunks."""
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().upper()
    print(f"\nCreating vector database using {provider} embeddings...")
    embeddings = get_embeddings()
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    print(f"[OK] Vector database created in '{VECTOR_DB_DIR}'")
    return vectorstore


def load_vector_store():
    """Load existing vector database if present."""
    if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
        print(f"Loading existing vector database from '{VECTOR_DB_DIR}'...")
        try:
            embeddings = get_embeddings()
        except Exception as e:
            print(f"Notice: Could not load embeddings: {e}")
            return None
            
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings
        )
        print("[OK] Vector database loaded")
        return vectorstore
    return None


def create_qa_chain(vectorstore):
    """Create a question-answering chain."""
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().upper()
    print(f"\nSetting up Q&A chain using {provider}...")
    llm = get_llm()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": NUM_RETRIEVED_DOCS}
        ),
        return_source_documents=True
    )
    print("[OK] Q&A chain ready")
    return qa_chain


def ask_question(qa_chain, question):
    """Ask a question and get an answer with sources."""
    print(f"\nQuestion: {question}")
    print("Searching documents...")
    try:
        if hasattr(qa_chain, "invoke"):
            result = qa_chain.invoke({"query": question})
        else:
            result = qa_chain({"query": question})
            
        answer = result.get("result") or result.get("answer", "")
        sources = result.get("source_documents") or result.get("context", [])
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": []
        }


def main():
    """Main function to initialize and run the Q&A system."""
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().upper()
    print("=" * 60)
    print(f"🌾 Agri Crop Management Q&A System ({provider} Powered)")
    print("=" * 60)
    
    vectorstore = load_vector_store()
    if vectorstore is None:
        print("\nNo existing database found. Creating new one...")
        documents = load_documents(DOCUMENTS_DIR)
        if not documents:
            print(f"No documents found! Please add PDF files to '{DOCUMENTS_DIR}'.")
            return
        chunks = split_documents(documents)
        try:
            vectorstore = create_vector_store(chunks)
        except Exception as e:
            print(f"\nError creating vector store: {e}")
            return
    
    try:
        qa_chain = create_qa_chain(vectorstore)
    except Exception as e:
        print(f"\nError setting up Q&A chain: {e}")
        return
    
    print("\n" + "=" * 60)
    print("System ready! Ask your agricultural questions (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        try:
            question = input("\n👤 Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye! 👋")
            break
            
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
        if not question:
            continue
            
        result = ask_question(qa_chain, question)
        print("\n" + "-" * 60)
        print("🤖 Answer:")
        print("-" * 60)
        print(result["answer"])
        
        if result["sources"]:
            print("\n" + "-" * 60)
            print("📚 Sources:")
            print("-" * 60)
            for i, source in enumerate(result["sources"][:3], 1):
                page = source.metadata.get('page', 'N/A')
                source_name = source.metadata.get('source', 'Unknown')
                print(f"{i}. Page {page} from {os.path.basename(source_name)}")
        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
