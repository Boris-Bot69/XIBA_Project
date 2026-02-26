import json
import os
from typing import List, Optional
from langchain_core.tools import tool
import numpy as np
from numpy.linalg import norm

from llm_utils import get_embedding_function
from utils import get_cwd

# Global variables to hold our in-memory store so it only loads once
_vector_store = None
_embeddings = None


def _initialize_vector_store():
    """Lazily initialize the vector store only when needed."""
    global _vector_store, _embeddings

    # If it's already loaded, skip initialization
    if _vector_store is not None:
        return

    print("Initializing case studies vector store...")
    _embeddings = get_embedding_function()

    file_path = os.path.join(get_cwd(), 'data', 'case_studies.json')

    with open(file_path, 'r') as f:
        case_studies = json.load(f)

    def build_text(case):
        return (
            f"Industry: {case['industry']}\n"
            f"Title: {case['title']}\n"
            f"C-Suite Summary: {case['c_suite_summary']}\n"
            f"Technical Summary: {case['technical_summary']}\n"
            f"Choice Rationale: {case['choice_rationale']}\n"
            f"Xibix Transformation: {case['xibix_transformation']}\n"
            f"Unique Approach: {case['unique_approach']}\n"
            f"Timeframe: {case['timeframe']}"
        )

    store = []
    for i, case in enumerate(case_studies):
        text = build_text(case)
        emb = _embeddings.embed_query(text)
        store.append({"id": i, "text": text, "embedding": np.array(emb, dtype=np.float32)})

    _vector_store = store
    print(f"Stored {len(_vector_store)} case studies in memory.")


@tool
def case_studies_tool(query: str, top_k: int = 3):
    """Access Xibix Case Studies and provide detailed case study based on the context, return top k results."""

    # Ensure the store is initialized before searching
    _initialize_vector_store()

    query_emb = np.array(_embeddings.embed_query(query), dtype=np.float32)
    scores = []

    for item in _vector_store:
        score = np.dot(query_emb, item["embedding"]) / (
                norm(query_emb) * norm(item["embedding"])
        )
        scores.append((score, item))

    results = sorted(scores, key=lambda x: x[0], reverse=True)[:top_k]
    print("\n\ncase study results >>>> ", results)

    return results