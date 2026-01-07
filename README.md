# RAG-chatbot-
Developed a RAG chatbot for interactive Q&amp;A on research papers, improving comprehension and assistance.

Below is a **complete, clean, production-quality `README.md`** you can directly copy into your repo.
It explains **both ways of using your project**:

1. **CLI / local execution using `test_graph.py`** (no Streamlit)
2. **Interactive UI using `app.py` (Streamlit)**
3. **Optional graph visualization using `visualize_graph.py`**

It is written so **any developer can clone and run it without asking you questions**.

---

# 📚 Graph RAG (Graph-based Retrieval Augmented Generation)

This repository implements a **Graph-based Retrieval Augmented Generation (Graph RAG)** system with a **hybrid Graph-first + Vector (FAISS) fallback strategy**.

Unlike traditional RAG systems that rely only on vector similarity, this project:

* Builds a **knowledge graph** from uploaded PDFs
* Performs **graph-based reasoning first**
* Falls back to **FAISS vector search only when necessary**
* Supports **multi-document reasoning**
* Provides **explainable retrieval paths**

The system can be used in **two ways**:

1. **Local / CLI execution** using `test_graph.py`
2. **Interactive UI** using `Streamlit` (`app.py`)

---

## 🚀 Features

* 📄 Upload one or multiple PDFs
* 🧠 Entity & relationship extraction using LLMs
* 🕸️ Knowledge graph construction
* 🔍 Graph-based retrieval (deterministic & explainable)
* 🔁 FAISS fallback for semantic recall
* 💬 Multi-turn chat interface (Streamlit)
* 📊 Optional graph visualization
* 🧪 Standalone testing without UI

---

## 📁 Project Structure

```
.
├── app.py                # Streamlit Graph RAG application
├── test_graph.py         # Run Graph RAG locally (no UI)
├── graph_util.py         # Graph construction & traversal logic
├── rag_util.py           # PDF loading, chunking, FAISS utilities
├── model.py              # LLM wrapper (HuggingFace Inference API)
├── visualize_graph.py    # Optional graph visualization (PyVis)
├── requirements.txt      # Python dependencies
├── README.md             # Documentation
├── .env.example          # Environment variable template
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/pal14920251/RAG-using-Graphs.git
cd RAG-using-Graphs
```

---

### 2️⃣ Create and activate a virtual environment (recommended)

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Set environment variables

This project uses the **Hugging Face Inference API**.

Create a `.env` file in the root directory:

```
HF_API_TOKEN=your_huggingface_api_token_here
```

> ⚠️ Do **NOT** commit `.env` to GitHub
> Use `.env.example` as a reference

---

## 🧪 Option 1: Run Graph RAG Locally (No UI)

Use this mode if you want to:

* Test Graph RAG logic
* Debug graph construction
* See reasoning paths in the terminal
* Experiment without Streamlit

### ▶️ Run

```bash
python test_graph.py
```

### 🔍 What `test_graph.py` does

1. Loads PDFs from a local path
2. Splits PDFs into chunks
3. Builds a **knowledge graph**
4. Builds a **FAISS vector index**
5. Runs **Graph-first retrieval**
6. Falls back to FAISS if needed
7. Prints:

   * Graph stats
   * Extracted entities
   * Reasoning paths
   * Final answer

This is ideal for **learning, debugging, and experimentation**.

---

## 🌐 Option 2: Run Interactive Streamlit App

Use this mode if you want:

* A frontend UI
* Easy PDF uploads
* Multi-turn chat
* Shareable demo

### ▶️ Run

```bash
streamlit run app.py
```

### 🧠 App Workflow

1. Upload one or more PDFs
2. Click **Build / Rebuild Graph**
3. Ask questions in chat format
4. Graph is reused across questions
5. FAISS is used only when graph context is insufficient
6. View reasoning paths per answer

---

## 📊 Optional: Visualize the Knowledge Graph

If you want to **visually inspect the graph**, you can use `visualize_graph.py`.

### Install visualization dependencies (if not already installed)

```bash
pip install networkx pyvis
```

### Usage

Inside `test_graph.py` (or any script), call:

```python
from visualize_graph import visualize_graph

visualize_graph(
    graph,
    output_file="graph.html",
    max_nodes=30
)
```

Then open `graph.html` in your browser to explore:

* Nodes (Entities / Concepts / Chunks)
* Relationships
* Cross-document links

> ⚠️ Recommended only for **small graphs** (debugging / learning)

---

## 🧠 How Retrieval Works (Important)

### Graph-First Strategy

* Extract entities from the user query
* Traverse the knowledge graph
* Retrieve grounded chunks

### FAISS Fallback

FAISS is used **only if**:

* Graph returns no chunks
* Or graph context is too small

This ensures:

* High precision
* Low hallucination
* Explainable answers

---

## 🔐 Notes on Persistence

* Graphs are **session-based**
* Graph is rebuilt only when:

  * New PDFs are uploaded
  * User clicks “Rebuild Graph”
* Graph is **not stored on disk** by default

(Designed intentionally for learning & safety)

---

## 🛠️ Requirements

* Python 3.9+
* Internet connection (for model inference)
* Hugging Face API token

---

## 🧩 Future Enhancements (Planned / Possible)

* Persistent graph storage (Neo4j / disk cache)
* Multi-hop graph traversal
* Source citations (PDF + page)
* Confidence scoring
* Toggle Graph-only vs Hybrid
* Deployment on Streamlit Cloud

---

## 📜 License

This project is intended for **educational and research purposes**.

---

## 🙌 Acknowledgements

Inspired by modern research on:

* Graph RAG
* Knowledge Graphs
* Hybrid retrieval systems

---

If you want, next I can:

* Review this README for **open-source best practices**
* Add **example screenshots**
* Prepare a **Streamlit Cloud deployment guide**
* Help you write a **blog or LinkedIn post** explaining this project

Just tell me 👍
