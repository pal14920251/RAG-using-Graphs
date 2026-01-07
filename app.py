import os
import streamlit as st
import rag_util
from model import ChatModel
from graph_util import (
    GraphExtractor,
    build_graph_from_chunks,
    QueryEntityExtractor,
    GraphRetriever,
    build_context_from_chunks
)

# =====================================================
# App Configuration
# =====================================================
st.set_page_config(
    page_title="Graph RAG Assistant",
    layout="wide"
)

st.title("📚 Graph RAG Assistant")
st.caption("Graph-first reasoning with FAISS fallback")

FILES_DIR = "files"
os.makedirs(FILES_DIR, exist_ok=True)

# =====================================================
# Session State Initialization
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph" not in st.session_state:
    st.session_state.graph = None

if "faiss_db" not in st.session_state:
    st.session_state.faiss_db = None

# =====================================================
# Load LLM (once)
# =====================================================
@st.cache_resource
def load_model():
    return ChatModel(model_id="deepseek-ai/DeepSeek-R1")

model = load_model()

# =====================================================
# Helper: Save uploaded PDFs
# =====================================================
def save_file(uploaded_file):
    path = os.path.join(FILES_DIR, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

# =====================================================
# Sidebar: Upload PDFs
# =====================================================
with st.sidebar:
    st.header("📄 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    rebuild = st.button("🔄 Build / Rebuild Graph")

# =====================================================
# Build Graph + FAISS (ONLY when user clicks rebuild)
# =====================================================
if rebuild and uploaded_files:
    with st.spinner("Building graph and vector index..."):

        file_paths = [save_file(f) for f in uploaded_files]

        # Load & split PDFs
        docs = rag_util.load_and_split_pdfs(
            file_paths,
            chunk_size=256
        )

        # Build FAISS index
        encoder = rag_util.Encoder(
            model_name="sentence-transformers/all-MiniLM-L12-v2",
            device="cpu"
        )

        faiss_db = rag_util.FaissDb(
            docs=docs,
            embedding_function=encoder.embedding_function
        )

        # Build Knowledge Graph
        extractor = GraphExtractor(llm=model)
        graph = build_graph_from_chunks(docs, extractor)

        # Store in session
        st.session_state.graph = graph
        st.session_state.faiss_db = faiss_db
        st.session_state.messages = []  # reset chat on rebuild

        st.success("✅ Graph & FAISS index built successfully")

# =====================================================
# Chat Interface
# =====================================================
if st.session_state.graph is not None:

    # -------------------------------
    # Display chat history
    # -------------------------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # -------------------------------
    # Chat input (persistent)
    # -------------------------------
    user_question = st.chat_input(
        "Ask a question about the uploaded PDFs"
    )

    if user_question:
        # Store user message
        st.session_state.messages.append(
            {"role": "user", "content": user_question}
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                graph = st.session_state.graph
                faiss_db = st.session_state.faiss_db

                # ==========================
                # PHASE 3: Graph RAG
                # ==========================
                qe = QueryEntityExtractor(llm=model)
                query_entities = qe.extract(
                    user_question,
                    graph=graph
                )

                retriever = GraphRetriever(graph)
                graph_chunk_ids = retriever.retrieve_chunks(
                    query_entities
                )

                graph_context = build_context_from_chunks(
                    graph,
                    graph_chunk_ids
                )

                # ==========================
                # Hybrid Graph + FAISS fallback
                # ==========================
                MIN_GRAPH_CHUNKS = 1
                MIN_CONTEXT_CHARS = 400

                use_faiss = (
                    len(graph_chunk_ids) < MIN_GRAPH_CHUNKS or
                    len(graph_context) < MIN_CONTEXT_CHARS
                )

                final_context = graph_context

                if use_faiss:
                    faiss_context = faiss_db.similarity_search(
                        user_question,
                        k=3
                    )
                    final_context = graph_context + "\n\n" + faiss_context

                # ==========================
                # LLM Answer
                # ==========================
                answer = model.generate(
                    question=user_question,
                    context=final_context,
                    max_new_tokens=300
                )

                st.markdown(answer)

        # Store assistant message
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        # -------------------------------
        # Explainability
        # -------------------------------
        with st.expander("🔍 Graph Reasoning Path"):
            for s, r, o in graph.edges:
                if s in query_entities:
                    st.write(f"{s} → {r} → {o}")

else:
    st.info(
        "⬅️ Upload PDFs and click **Build / Rebuild Graph** to start."
    )
