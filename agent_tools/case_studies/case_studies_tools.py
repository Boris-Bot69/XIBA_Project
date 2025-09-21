import json
from datetime import datetime
import os
from typing import List, Optional
from langchain_core.tools import tool
import numpy as np
from numpy.linalg import norm

from llm_utils import get_embedding_function
from utils import get_cwd



# Build in-memory vector store
case_studies = []
embeddings = get_embedding_function()

with open(get_cwd() + os.sep + 'data' + os.sep + 'case_studies.json', 'r') as f:
    case_studies = json.load(f)

# Decide what fields to use for embeddings
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

# Build in-memory vector store
vector_store = []
for i, case in enumerate(case_studies):
    text = build_text(case)
    emb = embeddings.embed_query(text)
    vector_store.append({"id": i, "text": text, "embedding": np.array(emb, dtype=np.float32)})

print(f"Stored {len(vector_store)} case studies in memory.")



@tool
def case_studies_tool(query, top_k=3):
    """Access Xibix Case Studies and provide detailed case study based on the context, return top k results."""

    query_emb = np.array(embeddings.embed_query(query), dtype=np.float32)
    scores = []
    for item in vector_store:
        score = np.dot(query_emb, item["embedding"]) / (
            norm(query_emb) * norm(item["embedding"])
        )
        scores.append((score, item))
    results = sorted(scores, key=lambda x: x[0], reverse=True)[:top_k]
    print ("\n\ncase study results >>>> ", results)
    return results