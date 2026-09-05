# 📄 PDF Chat & Chunking Studio

> **An interactive RAG experimentation platform for exploring, analyzing, and comparing text chunking strategies for PDF documents.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🚀 Overview

**PDF Chat & Chunking Studio** is an interactive Streamlit application designed to help developers and AI/ML learners understand how **document chunking affects Retrieval-Augmented Generation (RAG)**.

Instead of treating a PDF chatbot as a black box, this project provides an interactive environment where you can:

* Upload and analyze PDF documents
* Experiment with different chunking strategies
* Adjust chunk size and overlap
* Visualize chunk distributions
* Compare two chunking configurations side by side
* Inspect chunk boundaries directly on document text
* Search and explore generated chunks
* Chat with documents using RAG
* Switch between local and LLM-powered retrieval modes
* Export chunks and conversations for further analysis

The goal is to make the **document processing and retrieval pipeline observable and explainable**.

---

## ✨ Features

### 🧩 Multi-Strategy Chunking

Experiment with four different approaches:

| Strategy                 | Description                                                                        |
| ------------------------ | ---------------------------------------------------------------------------------- |
| **Recursive Character**  | Splits text hierarchically using paragraphs, newlines, spaces, and characters.     |
| **Semantic Splitting**   | Detects topic transitions using sentence-level TF-IDF vectors and cosine distance. |
| **Sentence-Based**       | Creates chunks around grammatical sentence boundaries.                             |
| **Fixed-Size Character** | Creates fixed-length character windows with configurable overlap.                  |

---

### 📊 Interactive Chunking Analytics

Explore how your chosen configuration affects the document:

* Total pages
* Total chunks
* Average chunk size
* Chunk length distribution
* Searchable chunk explorer
* Page-level source information
* JSON export
* CSV export

---

### 🔬 Chunking Strategy Comparison

The **Comparison Studio** allows you to run two configurations against the same document.

You can compare:

* Different chunking strategies
* Different chunk sizes
* Different overlap values
* Chunk length distributions
* Chunk boundaries
* Resulting chunk structures

This makes it easier to understand how preprocessing decisions can affect a RAG system.

---

### 💬 RAG Chat Studio

Chat with your uploaded documents using multiple retrieval modes.

#### 📴 Local TF-IDF

A completely local retrieval mode using:

* TF-IDF vectorization
* Cosine similarity
* In-memory retrieval

No API key or external LLM service is required.

#### 🤖 OpenAI

Supports:

* `text-embedding-3-small`
* GPT-4o
* GPT-4o-mini

#### 🦙 Ollama

Connect the application to a local Ollama installation and use locally hosted open-weight models.

---

### 🎛️ Advanced RAG Controls

Configure the retrieval pipeline using:

* Top-K retrieval
* Similarity score threshold
* Custom system prompts
* `{context}` prompt placeholder
* Source page information
* Similarity scores
* Conversation history export

---

### 🔎 Source Traceability

Retrieved answers include source information such as:

* Document page numbers
* Retrieved passages
* Similarity scores

This provides greater transparency into **where the generated answer came from**.

---

## 🧠 How It Works

The application follows a simplified RAG pipeline:

```text
                 ┌─────────────────┐
                 │    PDF Upload   │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  PDF Extraction │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Chunking Engine │
                 └────────┬────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
         Recursive     Semantic    Sentence /
         Character     Splitting   Fixed Size
              │           │           │
              └───────────┼───────────┘
                          ▼
                 ┌─────────────────┐
                 │ Chunk Analytics │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Vector Indexing │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ User Question   │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Similarity      │
                 │ Retrieval       │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Context         │
                 │ Construction    │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ LLM / Response  │
                 └─────────────────┘
```

---

## 🧩 Chunking Pipeline

### 1. Recursive Character Splitting

The text is recursively divided using a hierarchy of separators:

```text
Paragraph
   ↓
Newline
   ↓
Space
   ↓
Character
```

This attempts to preserve natural text structure while maintaining the desired chunk size.

### 2. Semantic Splitting

The semantic splitter:

1. Tokenizes the document into sentences.
2. Creates TF-IDF representations.
3. Calculates cosine distance between adjacent sentences.
4. Detects significant topic transitions.
5. Creates chunk boundaries around those transitions.

### 3. Sentence-Based Splitting

Text is divided using sentence-level punctuation and then grouped into chunks according to the configured target size.

### 4. Fixed-Size Splitting

The document is divided into character windows using:

```text
step = chunk_size - chunk_overlap
```

This provides predictable chunk sizes and configurable overlap.

---

## 🔍 Retrieval Pipeline

After chunking:

```text
PDF
 ↓
Text Chunks
 ↓
Vector Representation
 ↓
Similarity Search
 ↓
Top-K Relevant Chunks
 ↓
Context
 ↓
Response Generation
```

### Local Retrieval

The Local TF-IDF mode performs retrieval entirely within the application using cosine similarity.

### LLM-Based Retrieval

OpenAI and Ollama modes use embeddings to identify relevant chunks before passing the retrieved context to the language model.

---

## 🖥️ Application Interface

The application is organized into three main areas:

### 📈 Analysis Studio

Analyze documents and inspect:

* Chunk statistics
* Chunk size distribution
* Chunk contents
* Source pages
* Exportable data

### ⚖️ Comparison Studio

Compare two chunking configurations and visually inspect their differences.

### 💬 Chat Studio

Ask questions about your document using the configured RAG backend.

---

## 📁 Project Structure

```text
pdf-chat-and-chunking-studio/
│
├── core/
│   ├── chunker.py
│   └── rag_engine.py
│
├── ui/
│   └── ui_components.py
│
├── views/
│   ├── __init__.py
│   ├── analysis_view.py
│   ├── chat_view.py
│   └── comparison_view.py
│
├── app.py
├── style.css
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

### Core Components

**`core/chunker.py`**

Handles:

* PDF text extraction
* Document processing
* Chunk generation
* Multiple chunking strategies

**`core/rag_engine.py`**

Handles:

* Vectorization
* Similarity search
* Retrieval
* RAG response generation
* Local and external model integrations

**`ui/ui_components.py`**

Contains reusable UI components and visual elements.

**`views/analysis_view.py`**

Responsible for document analysis and chunking analytics.

**`views/comparison_view.py`**

Handles chunking strategy comparison and boundary visualization.

**`views/chat_view.py`**

Handles the RAG chat interface and conversation functionality.

**`app.py`**

Acts as the main Streamlit application entry point and coordinates the different components.

---

## 🛠️ Tech Stack

| Technology       | Purpose                       |
| ---------------- | ----------------------------- |
| **Python**       | Application and RAG logic     |
| **Streamlit**    | Interactive web interface     |
| **PyMuPDF**      | PDF text extraction           |
| **NumPy**        | Numerical operations          |
| **Pandas**       | Data processing and analytics |
| **Scikit-learn** | TF-IDF and cosine similarity  |
| **OpenAI API**   | Embeddings and LLM responses  |
| **Ollama**       | Local LLM integration         |
| **Altair**       | Data visualization            |
| **CSS**          | Custom application styling    |

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Hatim-52/pdf-chat-and-chunking-studio.git
cd pdf-chat-and-chunking-studio
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Application

```bash
python -m streamlit run app.py --server.port 8504
```

Open your browser and visit:

```text
http://localhost:8504
```

---

## 🔑 API Configuration

The application can operate locally using **TF-IDF retrieval**, so an external API key is not required for basic document analysis and retrieval.

For OpenAI-powered RAG, configure your API key through an environment variable or Streamlit secrets.

Example:

```text
OPENAI_API_KEY=your_api_key_here
```

> ⚠️ Never commit API keys, `.env` files, or Streamlit secrets to GitHub.

---

## 🦙 Ollama Setup

To use local Ollama models:

1. Install Ollama.
2. Start the Ollama service.
3. Download a supported model.
4. Select **Ollama** from the application settings.

This allows RAG generation using a locally hosted language model.

---

## 📤 Export Options

The application supports exporting:

### Chunk Data

```text
CSV
JSON
```

### Chat History

```text
JSON
```

This makes it possible to use the generated data for additional experimentation or evaluation.

---

## 🎯 Why This Project?

Traditional PDF chat applications often hide the document-processing pipeline behind a simple chat interface.

This project focuses on making the **RAG preprocessing and retrieval pipeline visible and experimentally useful**.

The main question behind the project is:

> **How does the way we chunk a document affect what a RAG system retrieves?**

By allowing users to change chunking strategies and compare their results, the application provides a practical environment for understanding one of the most important stages of a RAG pipeline.

---

## 🚀 Future Improvements

Potential improvements include:

* [ ] RAG evaluation metrics
* [ ] Retrieval precision and recall analysis
* [ ] Context relevance scoring
* [ ] Answer faithfulness evaluation
* [ ] Retrieval latency tracking
* [ ] Automated chunking recommendations
* [ ] Support for additional document formats
* [ ] Persistent vector databases
* [ ] Batch document processing
* [ ] RAG experiment history
* [ ] Automated evaluation datasets
* [ ] Docker deployment
* [ ] Cloud deployment

---

## 📌 Learning Outcomes

This project demonstrates practical experience with:

* Retrieval-Augmented Generation
* Natural Language Processing
* Text preprocessing
* Document chunking
* Vector similarity search
* TF-IDF
* Embeddings
* LLM integration
* Streamlit application development
* Data visualization
* Modular Python architecture
* Local AI/LLM workflows

---

## 📄 License

This project is licensed under the **MIT License**.

See the [LICENSE] file for details.

---

## 👨‍💻 Author

**Hatim Noor**

---