# matcher_transformers.py – SBERT rerank (SentenceTransformers)
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import List, Tuple
import torch
from sentence_transformers import SentenceTransformer, util

def rerank(query: str, candidate_df, top_k: int = 10, model: SentenceTransformer = None) -> List[Tuple[str, str, float]]:
    if candidate_df is None or len(candidate_df) == 0:
        return []

    # Use provided model; otherwise load one (once)
    sbert = model if model is not None else SentenceTransformer("all-MiniLM-L6-v2")

    # Encode
    q_emb = sbert.encode(query, convert_to_tensor=True)
    cand_texts = candidate_df["text"].tolist()
    cand_embs = sbert.encode(cand_texts, convert_to_tensor=True)

    # Cosine similarity & top-k
    cos_scores = util.cos_sim(q_emb, cand_embs)[0]  # (N,)
    top_results = torch.topk(cos_scores, k=min(top_k, len(cand_texts)))

    out = []
    ids = candidate_df["id"].tolist()
    texts = candidate_df["text"].tolist()
    clusters = candidate_df["cluster"].tolist()
    for score, idx in zip(top_results.values, top_results.indices):
        i = int(idx)
        out.append((ids[i], texts[i], float(score), clusters[i]))
    return out
