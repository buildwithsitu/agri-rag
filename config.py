import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Provider selection: "gemini" (free tier) or "openai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Directory Configuration
DOCUMENTS_DIR = "documents"
VECTOR_DB_DIR = "agri_db"

# Chunking Configuration
CHUNK_SIZE = 1000  # Number of characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

# Retrieval Configuration
NUM_RETRIEVED_DOCS = 3  # Number of document chunks to retrieve

# Model Configuration
TEMPERATURE = 0.2  # Low temperature for accurate, grounded agricultural answers

# Gemini models (Free tier available at https://aistudio.google.com/app/apikey)
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"

# OpenAI models
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
