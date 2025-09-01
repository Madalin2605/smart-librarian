# Smart Librarian

Smart Librarian is an **AI-powered book recommendation chatbot** built with Retrieval-Augmented Generation (RAG), text-to-speech, and image generation.  
It provides personalized book suggestions and an interactive interface using via both **Streamlit UI** and a **CLI interface**.

---

## Project Structure
```
smart_librarian/
├── chatbot/                 # Core chatbot logic (agent, retriever, interface)
│   ├── agent.py             # Manages AI interactions
│   ├── interface.py         # Handles chatbot responses and user input
│   ├── retriever.py         # Retrieves context/books from database
│   └── __init__.py
│
├── tools/                   # Tools for chatbot
│   ├── image_generator.py   # Book cover generation with DALL·E
│   ├── language_filter.py   # Profanity filter / language checks
│   ├── summary_tool.py      # Summarization utilities
│   └── __init__.py
│
├── outputs/                 # Generated content
│   ├── audio/               # Text-to-speech audio files
│   └── images/              # AI-generated book covers
│
├── data/                    # Book datasets / resources
├── db/                      # ChromaDB vector database
│
├── streamlit_app.py         # Streamlit web interface entry point
├── CLI_app.py               # CLI interface entry point
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

```

---

## Features
- **Book Retrieval**: Search and recommend books using semantic search (ChromaDB).  
- **AI Chatbot**: Powered by OpenAI models (4o-mini, 4.1-mini, 4.1-nano).  
- **Image Generation**: Creates book cover art with DALL·E.  
- **Text-to-Speech**: Converts recommendations to audio using `pyttsx3`.
- **Profanity Filter**: Ensures clean and safe responses.  
- **CLI Interface**: Use the chatbot directly in your terminal.    
- **Streamlit UI**: Simple and interactive web interface.

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/smart-librarian.git
cd smart-librarian
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate          # On Linux/Mac
source .venv/Scripts/activate      # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set environment variable
Export your OpenAI API key:
```bash
export OPENAI_API_KEY="your_api_key_here"   # On Linux/Mac
setx OPENAI_API_KEY "your_api_key_here"     # On Windows
```

---

## Usage

### Run the Streamlit app (Web UI)
```bash
streamlit run streamlit_app.py
```

### Run the CLI version
```bash
python CLI_app.py
```

This allows interaction with the chatbot directly in your terminal.

---

## Tech Stack
- **Python 3.9+**
- **Streamlit** (frontend)
- **ChromaDB** (vector store)
- **OpenAI API** (LLM + DALL·E)
- **pyttsx3** (text-to-speech)
- **Profanity filter** (for language checks)