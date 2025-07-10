# explain_fast.py â€“ simple, instant, always informative
import numpy as np

class Explainer:
    def __init__(self, vectorizer, tfidf_matrix):
        self.vectorizer = vectorizer
        self.tfidf      = tfidf_matrix.toarray()            # shape (N_docs, V)
        self.tokens     = vectorizer.get_feature_names_out()

    # ------------------------------------------------------------------

    def explain(self, query_text: str, doc_idx: int, top_k: int = 15):
        """
        Deterministic token-level contribution.
        Returns (top_k tokens, their contributions) for one document.
        """
        q_vec = self.vectorizer.transform([query_text]).toarray()[0]   # (V,)
        d_vec = self.tfidf[doc_idx]                                    # (V,)

        # Element-wise product approximates cosine numerator
        contrib = q_vec * d_vec                                        # (V,)

        # Indices of top-k absolute contributions
        idx = np.argsort(np.abs(contrib))[::-1][:top_k]
        return self.tokens[idx], contrib[idx]
