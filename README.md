# 🌾 Agri Crop Management Q&A System

A Retrieval-Augmented Generation (RAG) system for agricultural crop management, disease prevention, and pest control.

## 📁 Project Structure

```
agri-rag/
│
├── documents/                  # Agricultural PDF knowledge base
│   ├── cotton_pest_control.pdf
│   ├── tomato_disease_prevention.pdf
│   └── wheat_cultivation_guide.pdf
│
├── agri_db/                    # Local Chroma vector database
├── venv/                       # Python virtual environment
├── .env                        # OpenAI API key configuration
├── .env.example                # Example configuration template
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Python dependencies
├── config.py                   # RAG parameters and model configuration
├── main.py                     # CLI Q&A interface & vector store builder
├── app.py                      # Streamlit web user interface
└── AGRI.md                     # Full guide & reference documentation
```

## 🚀 Quick Start

### 1. Activate Virtual Environment
In PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Configure API Key
Open `.env` and set your OpenAI API key:
```env
OPENAI_API_KEY=sk-your-actual-api-key
```

### 3. Run the CLI Application
Build the vector database and ask questions from the command line:
```powershell
python main.py
```

### 4. Run the Streamlit Web Application
Launch the interactive web UI:
```powershell
streamlit run app.py
```

## 🌾 Example Questions
- "What are common cotton pests and how do I manage them?"
- "How do I prevent fungal diseases in tomatoes?"
- "What is the optimal sowing time and irrigation schedule for wheat?"

