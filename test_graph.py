# import rag_util
# from graph_util import GraphExtractor, build_graph_from_chunks, QueryEntityExtractor, GraphRetriever, build_context_from_chunks
# from model import ChatModel
# from visualize_graph import visualize_graph

# # 1. Load model
# model = ChatModel(model_id="deepseek-ai/DeepSeek-R1")

# # 2. Load and split PDF
# file_paths = [r"C:\Users\91760\Downloads\Plate tectonics.pdf"]

# docs = rag_util.load_and_split_pdfs(file_paths, chunk_size=256)

# print("Chunks created:", len(docs))

# # 3. Build graph
# extractor = GraphExtractor(llm=model)
# graph = build_graph_from_chunks(docs, extractor)

# # 4. Inspect graph
# print("Total Nodes:", len(graph.nodes))
# print("Total Edges:", len(graph.edges))

# # Show some nodes
# print("\nSample Nodes:")
# for i, (k, v) in enumerate(graph.nodes.items()):
#     print(k, "->", v)
#     if i == 5:
#         break

# # Show some edges
# print("\nSample Edges:")
# for edge in graph.edges[:10]:
#     print(edge)

# # Show entity → chunk links
# print("\nEntity to Chunks Mapping:")
# for entity, chunks in list(graph.chunk_entity_map.items())[:5]:
#     print(entity, "->", chunks)

# visualize_graph(graph, output_file="plate_tectonics_graph.html", max_nodes=30)

# import rag_util
# from graph_util import (
#     GraphExtractor,
#     build_graph_from_chunks,
#     QueryEntityExtractor,
#     GraphRetriever,
#     build_context_from_chunks
# )
# from model import ChatModel
# from visualize_graph import visualize_graph

# # -------------------------------
# # 1. Load LLM
# # -------------------------------
# model = ChatModel(model_id="deepseek-ai/DeepSeek-R1")

# # -------------------------------
# # 2. Load and split PDF
# # -------------------------------
# file_paths = [r"C:\Users\91760\Downloads\Plate tectonics.pdf"]

# docs = rag_util.load_and_split_pdfs(file_paths, chunk_size=256)
# print("Chunks created:", len(docs))


# #a build FAISS DB
# encoder = rag_util.Encoder(
#     model_name="sentence-transformers/all-MiniLM-L12-v2",
#     device="cpu"
# )

# faiss_db = rag_util.FaissDb(
#     docs=docs,
#     embedding_function=encoder.embedding_function
# )

# # -------------------------------
# # 3. Build Knowledge Graph (Phase-2)
# # -------------------------------
# extractor = GraphExtractor(llm=model)
# graph = build_graph_from_chunks(docs, extractor)

# print("\n--- GRAPH STATS ---")
# print("Total Nodes:", len(graph.nodes))
# print("Total Edges:", len(graph.edges))

# # -------------------------------
# # 4. Inspect Graph (sanity check)
# # -------------------------------
# print("\nSample Nodes:")
# for i, (k, v) in enumerate(graph.nodes.items()):
#     print(k, "->", v)
#     if i == 5:
#         break

# print("\nSample Edges:")
# for edge in graph.edges[:10]:
#     print(edge)

# print("\nEntity to Chunks Mapping:")
# for entity, chunks in list(graph.chunk_entity_map.items())[:5]:
#     print(entity, "->", chunks)

# # -------------------------------
# # 5. Visualize Graph (Optional but recommended)
# # -------------------------------
# visualize_graph(
#     graph,
#     output_file="plate_tectonics_graph.html",
#     max_nodes=30
# )

# # =====================================================
# # =============== PHASE 3 STARTS HERE =================
# # =====================================================

# print("\n================ GRAPH RAG QUERY =================")

# # -------------------------------
# # 6. User Question
# # -------------------------------
# user_question = "Why is plate tectonics considered a unifying theory?"

# print("User Question:", user_question)

# # -------------------------------
# # 7. Extract Query Entities
# # -------------------------------
# query_entity_extractor = QueryEntityExtractor(llm=model)
# query_entities = query_entity_extractor.extract(
#     user_question,
#     graph=graph   # IMPORTANT: graph-aware fallback
# )

# print("\nExtracted Query Entities:")
# print(query_entities)


# print("\n================ GRAPH REASONING PATH =================")
# for s, r, o in graph.edges:
#     if s in query_entities:
#         print(f"{s} --{r}--> {o}")

# # -------------------------------
# # 8. Graph-based Retrieval
# # -------------------------------
# retriever = GraphRetriever(graph)
# chunk_ids = retriever.retrieve_chunks(query_entities)
# context=build_graph_from_chunks(graph,chunk_ids)

# print("\nGraph Chunk IDs:", chunk_ids)
# print("Graph Context Length:", len(context))



# # -------------------------------
# # 10. Final Answer (Graph RAG)
# # -------------------------------
# answer = model.generate(
#     question=user_question,
#     context=context,
#     max_new_tokens=300
# )

# print("\n================ FINAL ANSWER =================")
# print(answer)


import rag_util
from graph_util import (
    GraphExtractor,
    build_graph_from_chunks,
    QueryEntityExtractor,
    GraphRetriever,
    build_context_from_chunks
)
from model import ChatModel
from visualize_graph import visualize_graph


# =====================================
# 1. Load LLM
# =====================================
model = ChatModel(model_id="deepseek-ai/DeepSeek-R1")


# =====================================
# 2. Load and split PDF
# =====================================
file_paths = [r"C:\Users\91760\Downloads\Plate tectonics.pdf"]

docs = rag_util.load_and_split_pdfs(file_paths, chunk_size=256)
print("Chunks created:", len(docs))


# =====================================
# 3. Build FAISS DB (for fallback)
# =====================================
encoder = rag_util.Encoder(
    model_name="sentence-transformers/all-MiniLM-L12-v2",
    device="cpu"
)

faiss_db = rag_util.FaissDb(
    docs=docs,
    embedding_function=encoder.embedding_function
)


# =====================================
# 4. Build Knowledge Graph (PHASE 2)
# =====================================
extractor = GraphExtractor(llm=model)
graph = build_graph_from_chunks(docs, extractor)

print("\n--- GRAPH STATS ---")
print("Total Nodes:", len(graph.nodes))
print("Total Edges:", len(graph.edges))


# =====================================
# 5. Inspect Graph (sanity check)
# =====================================
print("\nSample Nodes:")
for i, (k, v) in enumerate(graph.nodes.items()):
    print(k, "->", v)
    if i == 5:
        break

print("\nSample Edges:")
for edge in graph.edges[:10]:
    print(edge)

print("\nEntity to Chunks Mapping:")
for entity, chunks in list(graph.chunk_entity_map.items())[:5]:
    print(entity, "->", chunks)


# =====================================
# 6. Visualize Graph (optional but useful)
# =====================================
visualize_graph(
    graph,
    output_file="plate_tectonics_graph.html",
    max_nodes=30
)


# =====================================================
# ================== PHASE 3 ==========================
# =====================================================

print("\n================ GRAPH RAG QUERY =================")

# ---------------------------------
# 7. User Question
# ---------------------------------
user_question = "Why is plate tectonics considered a unifying theory of Earth sciences?"
print("User Question:", user_question)


# ---------------------------------
# 8. Extract Query Entities
# ---------------------------------
query_entity_extractor = QueryEntityExtractor(llm=model)
query_entities = query_entity_extractor.extract(
    user_question,
    graph=graph   # IMPORTANT: graph-aware fallback
)

print("\nExtracted Query Entities:")
print(query_entities)


# ---------------------------------
# 9. Graph Reasoning Path (Explainability)
# ---------------------------------
print("\n================ GRAPH REASONING PATH =================")
for s, r, o in graph.edges:
    if s in query_entities:
        print(f"{s} --{r}--> {o}")


# ---------------------------------
# 10. Graph-based Retrieval
# ---------------------------------
retriever = GraphRetriever(graph)
graph_chunk_ids = retriever.retrieve_chunks(query_entities)

graph_context = build_context_from_chunks(graph, graph_chunk_ids)

print("\nGraph Chunk IDs:", graph_chunk_ids)
print("Graph Context Length:", len(graph_context))


# ---------------------------------
# 11. HYBRID GRAPH + FAISS FALLBACK
# ---------------------------------
MIN_GRAPH_CHUNKS = 1
MIN_CONTEXT_CHARS = 400

use_faiss = (
    len(graph_chunk_ids) < MIN_GRAPH_CHUNKS or
    len(graph_context) < MIN_CONTEXT_CHARS
)

final_context = graph_context

if use_faiss:
    print("\n⚠️ Graph context insufficient — using FAISS fallback")

    faiss_context = faiss_db.similarity_search(
        user_question,
        k=3
    )

    final_context = graph_context + "\n\n" + faiss_context
else:
    print("\n✅ Graph context sufficient — FAISS not used")


print("\n--- FINAL CONTEXT SENT TO LLM (preview) ---")
print(final_context[:1000])


# ---------------------------------
# 12. Final Answer (Graph RAG)
# ---------------------------------
answer = model.generate(
    question=user_question,
    context=final_context,
    max_new_tokens=300
)

print("\n================ FINAL ANSWER =================")
print(answer)
