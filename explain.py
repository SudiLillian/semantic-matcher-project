"""
explain.py  —  WORKS with SHAP ≥ 0.42
--------------------------------------
Adds an Independent masker so we never see
“masker cannot be None”.
"""

import shap
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Explainer:
    def __init__(self, vectorizer, tfidf_matrix):
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix

        # --- NEW: explicit masker -----------------------------------------
        n_features = tfidf_matrix.shape[1]
        masker = shap.maskers.Independent(np.zeros((1, n_features)))
        # -------------------------------------------------------------------

        self.explainer = shap.Explainer(self._score, masker=masker)

    # model function passed to SHAP
    def _score(self, query_vec):
        # query_vec comes in dense; (m, d) → (m, N_docs)
        return cosine_similarity(query_vec, self.tfidf_matrix)

    def explain(self, query_text, doc_index, top_k=10):
        """
        Returns top-k tokens and their Shapley values for similarity between
        `query_text` and document at `doc_index`.
        """
        q_dense = self.vectorizer.transform([query_text]).toarray()
        shap_vals = self.explainer(q_dense)          # (1, N_docs, d)
        token_names = self.vectorizer.get_feature_names_out()

        doc_shap = shap_vals.values[0][doc_index]    # (d,)
        idx = np.argsort(np.abs(doc_shap))[::-1][:top_k]
        return token_names[idx], doc_shap[idx]

