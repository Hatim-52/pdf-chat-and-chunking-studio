# PDF Chat & Chunking Studio

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)

An interactive, user-friendly Streamlit web application that lets you upload PDF documents, explore different text-chunking parameters, view real-time chunking analytics, compare multiple chunking strategies side-by-side, and chat with your documents using Retrieval-Augmented Generation (RAG).

## 🚀 Features

- **Multi-Strategy Chunking**:
  - **Recursive Character splitting** (recommended): Splits by paragraphs, then sentences, then words.
  - **Semantic splitting**: Computes TF-IDF sentence distance and identifies natural topic transition boundaries.
  - **Sentence-based splitting**: Respects grammatical sentence boundaries (`.`, `!`, `?`).
  - **Fixed-size Character splitting**: Uniform character-length slices with overlap.
- **Interactive Visual Analytics**:
  - Live charts showing chunk length distribution.
  - Core statistics: Total Pages, Total Chunks, Average Chunk Size.
  - Searchable data table to inspect and filter individual chunks.
  - Download options to export chunks to CSV or JSON formats.
- **📊 Strategy Comparison Studio**:
  - Side-by-side comparison of two chunking configurations on the same document.
  - Overlay distribution histogram charts.
  - **Visual Chunk Alignment Inspector**: Color-coded HTML boundary highlights showing exactly where chunk boundaries occur.
- **💬 RAG Chat Studio & Advanced Settings**:
  - **Local TF-IDF mode**: Runs completely offline, zero-config. No API keys or external servers required!
  - **OpenAI mode**: Connects to OpenAI embeddings and GPT-4o / GPT-4o-mini models.
  - **Ollama mode**: Connects to a local running Ollama instance to use local open-weights LLMs.
  - **Configurable Top K & Similarity Score Threshold**: Filter out low-relevance passages.
  - **Custom System Prompt Template**: Inject tailored instructions or use `{context}` placeholders.
  - **Export Chat History**: Download conversation history in JSON format.
  - **Confirmation-Guarded Chat Clearing**: Popover-protected reset button.
- **Traceability / Source Citation**: View exactly which parts of the document were retrieved with calculated similarity scores and page numbers.
- **Premium UI & UX**: Modern typography (Inter font), custom metrics cards, responsive layouts, and smooth animations.

---

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/pdf-chat-chunking-studio.git
cd pdf-chat-chunking-studio
```

### 2. Install Dependencies
Make sure you have Python 3.10+ installed. In your terminal, run:
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Start the Streamlit development server:
```bash
python -m streamlit run app.py --server.port 8504
```

The application will be accessible in your web browser at:
`http://localhost:8504`

---

## 📂 File Structure

```
├── app.py                  # Main lightweight orchestrator (~85 lines)
├── style.css               # Clean stylesheet
├── ui_components.py        # Reusable UI cards, boundary highlights & citations
├── views/
│   ├── __init__.py
│   ├── analysis_view.py    # Tab 1: Upload, Chunking & Live Analytics
│   ├── comparison_view.py  # Tab 2: Strategy Comparison & Visual Inspector
│   └── chat_view.py        # Tab 3: RAG Chat Studio & History Export
├── chunker.py              # Text extraction (PyMuPDF) & 4 chunking strategies
├── rag_engine.py           # Vector search & RAG generation pipelines
├── requirements.txt        # Python package dependencies
├── LICENSE                 # MIT License
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

---

## 💡 How It Works

### Chunking Strategies
1. **Recursive Character**: Uses a hierarchy of separators (`\n\n`, `\n`, ` `, `""`) to split text into chunks while preserving paragraph and sentence integrity.
2. **Semantic**: Tokenizes sentences, builds term frequency vectors, computes cosine distances between adjacent sentences, and places boundaries at points of high topic divergence.
3. **Sentence-based**: Splits by punctuation boundaries (`.`, `!`, `?`) and aggregates sentences up to the specified chunk size with overlap.
4. **Fixed-size Character**: Slices text into uniform blocks of fixed character length.

### Retrieval & Generation
- When a document is processed, a vector index is constructed in-memory.
- For **Local TF-IDF**, a Term Frequency-Inverse Document Frequency matrix is constructed using NumPy with cosine similarity matching and extractive sentence ranking.
- For **OpenAI** and **Ollama**, embeddings are generated for each chunk. Cosine similarity retrieves the top $K$ chunks filtered by the similarity threshold, which are then passed to the LLM.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


