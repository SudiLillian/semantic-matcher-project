# explain_fast.py – TF-IDF Explainer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class Explainer:
    """
    Token / n-gram contribution explainer using TF-IDF overlap.
    SHAP-style coloring: positive → green, negative → red.
    """

    def __init__(self, vectorizer=None, tfidf_matrix=None, ngram_range=(1, 2)):
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.ngram_range = ngram_range
        self.tokens = None

    def fit(self, docs):
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range)
        if self.tfidf_matrix is None:
            self.tfidf_matrix = self.vectorizer.fit_transform(docs).toarray()
        self.tokens = self.vectorizer.get_feature_names_out()
        return self

    def explain(self, query_text, doc_idx, top_k=15):
        query_vec = self.vectorizer.transform([query_text]).toarray()[0]
        doc_vec = self.tfidf_matrix[doc_idx]
        contributions = query_vec * doc_vec
        mask = contributions != 0
        if not mask.any():
            return [], [], []
# explain_fast.py – TF-IDF Explainer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class Explainer:
    """
    Token / n-gram contribution explainer using TF-IDF overlap.
    SHAP-style coloring: positive → green, negative → red.
    """

    def __init__(self, vectorizer=None, tfidf_matrix=None, ngram_range=(1, 2)):
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.ngram_range = ngram_range
        self.tokens = None

    def fit(self, docs):
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range)
        if self.tfidf_matrix is None:
            self.tfidf_matrix = self.vectorizer.fit_transform(docs).toarray()
        self.tokens = self.vectorizer.get_feature_names_out()
        return self

    def explain(self, query_text, doc_idx, top_k=15):
        query_vec = self.vectorizer.transform([query_text]).toarray()[0]
        doc_vec = self.tfidf_matrix[doc_idx]
        contributions = query_vec * doc_vec
        mask = contributions != 0
        if not mask.any():
            return [], [], []

        contributions = contributions[mask]
        tokens_nz = self.tokens[mask]
        top_indices = np.argsort(np.abs(contributions))[::-1][:top_k]

        top_tokens = tokens_nz[top_indices]
        top_contribs = contributions[top_indices]

        # normalize [-1,1]
        norm_contribs = top_contribs / np.max(np.abs(top_contribs))

        # generate SHAP-style colors
        colors = [
            f"rgb(0,{int(255*val)},0)" if val>0 else f"rgb({int(255*abs(val))},0,0)"
            for val in norm_contribs
        ]
        return top_tokens, norm_contribs, colors

        contributions = contributions[mask]
        tokens_nz = self.tokens[mask]
        top_indices = np.argsort(np.abs(contributions))[::-1][:top_k]

        top_tokens = tokens_nz[top_indices]
        top_contribs = contributions[top_indices]

        # normalize [-1,1]
        norm_contribs = top_contribs / np.max(np.abs(top_contribs))

        # generate SHAP-style colors
        colors = [
            f"rgb(0,{int(255*val)},0)" if val>0 else f"rgb({int(255*abs(val))},0,0)"
            for val in norm_contribs
        ]
        return top_tokens, norm_contribs, colors
