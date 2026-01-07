from collections import defaultdict
from typing import List, Dict
import json
import re

class GraphStore:
    def __init__(self):
        self.nodes: Dict[str,dict]={}
        self.edges:List[tuple]=[]
        self.chunk_entity_map=defaultdict(list)
    def add_node(self,node_id:str,node_type:str,**metadata):
        if node_id not in self.nodes:
            self.nodes[node_id]={
                "type":node_type,
                **metadata
            }

    def add_edge(self,source:str,relation:str,target:str):
        self.edges.append((source,relation,target))


ALLOWED_RELATIONS={
    "MENTIONS",
    "DESCRIBES",
    "USED_FOR",
    "RELATED_TO",
    "PART_OF",
    "IMPROVES",
    "CAUSES"
}

EXTRACTION_PROMPT = """
You are an information extraction system.

Extract factual knowledge from the text below.

Rules:
- Only extract facts explicitly stated in the text.
- Use short, canonical names.
- Do NOT guess or infer.
- Use ONLY the allowed relations.
- Output VALID JSON ONLY (no explanation, no markdown).

Allowed relations:
MENTIONS, DESCRIBES, USED_FOR, RELATED_TO, PART_OF, IMPROVES, CAUSES

Output format:
[
  {{"subject": "...", "relation": "...", "object": "..."}}
]

Text:
{chunk}
"""


class GraphExtractor:
    def __init__(self, llm):
        self.llm = llm

    def extract_triples(self, chunk_text: str):
    
        prompt = EXTRACTION_PROMPT.format(chunk=chunk_text)
        response = self.llm.generate(prompt)
        print("RAM LLM OUTPUT:\n ",response)

        # 🔥 Extract JSON array from LLM output
        match = re.search(r"\[\s*{.*?}\s*\]", response, re.DOTALL)

        if not match:
            return []

        try:
            triples = json.loads(match.group())

            valid = []
            for t in triples:
                if (
                    isinstance(t, dict)
                    and "subject" in t
                    and "relation" in t
                    and "object" in t
                    and t["relation"] in ALLOWED_RELATIONS
                ):
                    valid.append(t)

            return valid

        except Exception:
            return []


def build_graph_from_chunks(chunks, extractor: GraphExtractor) -> GraphStore:
    """
    chunks: output of load_and_split_pdfs()
    extractor: GraphExtractor instance
    """

    graph = GraphStore()

    for idx, chunk in enumerate(chunks):
        print(f"Processing chunk {idx+1}/{len(chunks)}")
        chunk_id = f"chunk_{idx}"

        # Add chunk node
        graph.add_node(
            node_id=chunk_id,
            node_type="Chunk",
            text=chunk.page_content,
            source=chunk.metadata.get("source"),
            page=chunk.metadata.get("page")
        )

        # Extract triples from chunk
        triples = extractor.extract_triples(chunk.page_content)
        for triple in triples:
            subj = triple["subject"]
            rel = triple["relation"]
            obj = triple["object"]

            # Add entity & concept nodes
            graph.add_node(subj, "Entity")
            graph.add_node(obj, "Concept")

            # Add edges
            graph.add_edge(subj, rel, obj)
            graph.add_edge(chunk_id, "MENTIONS", subj)

            # Map entity to chunk
            graph.chunk_entity_map[subj].append(chunk_id)

    return graph


#now this is addition for fidning the entity to start from ,from the query asked by the user


QUERY_ENTITY_PROMPT = """
Extract key entities or concepts from the question.

Rules:
- Use short canonical names
- No explanations
- Output valid JSON only

Output format:
["entity1", "entity2", ...]

Question:
{question}
"""

class QueryEntityExtractor:
    def __init__(self, llm):
        self.llm = llm

    def extract(self, question: str, graph=None):
        prompt = QUERY_ENTITY_PROMPT.format(question=question)
        response = self.llm.generate(prompt)

        # 1️⃣ Try strict JSON
        try:
            entities = json.loads(response)
            if isinstance(entities, list) and entities:
                return entities
        except Exception:
            pass

        # 2️⃣ FALLBACK: string match from graph entities
        if graph:
            lowered_q = question.lower()
            matched = []
            for node, data in graph.nodes.items():
                if data["type"] == "Entity" and node.lower() in lowered_q:
                    matched.append(node)

            return matched

        return []

 

class GraphRetriever:
    def __init__(self, graph):
        self.graph = graph

    def retrieve_chunks(self, query_entities, max_hops=1):
        relevant_chunks = set()
        visited_entities = set(query_entities)

        for entity in query_entities:
            # Direct chunk grounding
            for chunk_id in self.graph.chunk_entity_map.get(entity, []):
                relevant_chunks.add(chunk_id)

            # Graph traversal (1-hop)
            for s, r, o in self.graph.edges:
                if s == entity and o not in visited_entities:
                    visited_entities.add(o)
                    for chunk_id in self.graph.chunk_entity_map.get(s, []):
                        relevant_chunks.add(chunk_id)

        return list(relevant_chunks)
def build_context_from_chunks(graph, chunk_ids, max_chars=2000):
    context = ""
    for cid in chunk_ids:
        text = graph.nodes[cid]["text"]
        context += text + "\n"

        if len(context) >= max_chars:
            break

    return context.strip()
