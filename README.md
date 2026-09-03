# PDF Chat & Chunking Studio

[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)

An interactive Streamlit web application that lets you upload PDF documents, explore and configure text chunking parameters, view real-time chunking analytics, compare multiple strategies side by side, and chat with your documents using Retrieval-Augmented Generation (RAG).

## Features

### Multi-Strategy Chunking
- Recursive Character Splitting: Uses hierarchical separators (paragraphs, newlines, spaces, characters) to maintain natural text flow.
- Semantic Splitting: Analyzes sentence term frequencies and cosine distances to split at topic transitions.
- Sentence-Based Splitting: Segments text along grammatical sentence boundaries.
- Fixed-Size Character Splitting: Chunks text into uniform lengths with customizable overlap.

### Interactive Analytics
- Real-time chunk character length distribution histogram.
- Key document statistics including total pages, total chunks, and average chunk size.
- Filterable and searchable chunk explorer table.
- Data export options to JSON and CSV formats.

### Strategy Comparison Studio
- Side-by-side comparison of two independent chunking configurations on the same document.
- Overlaid size distribution charts for visual comparison.
- Visual chunk alignment inspector displaying color-coded boundary highlights directly on page text.

### RAG Chat Studio and Search Options
- Local TF-IDF mode: Runs completely offline without API keys or external services.
- OpenAI mode: Integrates with text-embedding-3-small and GPT-4o / GPT-4o-mini models.
- Ollama mode: Connects to a local Ollama instance to run open-weights language models.
- Configurable top-k retrieval and similarity score thresholds.
- Custom system prompt templates with optional context placeholders.
- Export full conversation history as JSON.
- Protected clear-history action with confirmation popover.
- Source citations with similarity scores and page numbers for transparent answers.

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/pdf-chat-chunking-studio.git
cd pdf-chat-chunking-studio
```

### 2. Install Dependencies
Ensure Python 3.10 or higher is installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Start the Streamlit development server:
```bash
python -m streamlit run app.py --server.port 8504
```

Access the application in your browser at:
`http://localhost:8504`

## Project Structure

```
├── app.py                  # Main lightweight orchestrator
├── style.css               # Application stylesheet
├── ui_components.py        # Reusable metric cards, boundary visualizer, and citations
├── views/
│   ├── __init__.py
│   ├── analysis_view.py    # Document upload, chunking, and live analytics
│   ├── comparison_view.py  # Strategy comparison and visual boundary inspector
│   └── chat_view.py        # RAG chat studio and export options
├── chunker.py              # PDF extraction and chunking implementations
├── rag_engine.py           # In-memory vector indices and generation pipelines
├── requirements.txt        # Package dependencies
├── .gitignore              # Git ignore rules
└── README.md               # Documentation
```

## How It Works

### Chunking Logic
1. Recursive Character: Evaluates text against a sequence of separators (`\n\n`, `\n`, ` `, `""`) starting from the largest block. Backtracking preserves target overlap.
2. Semantic: Tokenizes sentences, builds term vectors, calculates cosine distance between adjacent sentences, and places boundaries where semantic distance exceeds the percentile threshold.
3. Sentence-Based: Breaks text by punctuation regex and joins sentences until the target chunk size is reached.
4. Fixed-Size: Slices character windows directly using step size `chunk_size - chunk_overlap`.

### Retrieval and Response Generation
- Document chunks are indexed into an in-memory vector space upon processing.
- In Local mode, TF-IDF vectors compute cosine similarity against user questions, returning ranked source passages and key sentence extractions.
- In OpenAI or Ollama modes, embeddings calculate cosine similarity to retrieve the top matching passages, which are then passed to the model prompt to generate grounded responses.
