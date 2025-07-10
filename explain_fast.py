# explain_fast.py  – simple, instant, always informative
import numpy as np

class Explainer:
    def __init__(self, vectorizer, tfidf_matrix):
        self.vectorizer = vectorizer
        self.tfidf = tfidf_matrix.toarray()                 # shape (N_docs, V)
        self.tokens = vectorizer.get_feature_names_out()

    def explain(self, query_text: str, doc_idx: int, top_k: int = 15):
        q_vec = self.vectorizer.transform([query_text]).toarray()[0]
        d_vec = self.tfidf[doc_idx]

        # Element-wise product = token-level cosine numerator
        contrib = q_vec * d_vec                             # shape (V,)

        # Get top-k absolute contributors (positive or negative)
        idx = np.argsort(np.abs(contrib))[::-1][:top_k]
        return self.tokens[idx], contrib[idx]
