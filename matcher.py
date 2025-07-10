import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import text

class Matcher:
    def __init__(self, profiles_df: pd.DataFrame):
        # store ids and texts
        self.ids   = profiles_df["id"].tolist()
        self.texts = profiles_df["text"].tolist()

        # custom stop-words
        my_stop_words = text.ENGLISH_STOP_WORDS.union(
            {"ai", "and", "the", "research", "researcher"}
        )

        # fit TF-IDF
        self.vectorizer = TfidfVectorizer(
            stop_words=my_stop_words,
            ngram_range=(1, 2),      # unigrams + bigrams
            max_features=5000,
            min_df=1,
            sublinear_tf=True,
            analyzer="word",
            token_pattern=r"(?u)\b\w[\w+-]*\b",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    # ------------------------------------------------------------------
    def match(self, query: str, top_n: int = 5):
        """Return top-N (id, text, score) ranked by cosine similarity."""
        q_vec  = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        ranked = sorted(
            zip(self.ids, self.texts, scores),
            key=lambda x: x[2],
            reverse=True,
        )
        return ranked[:top_n]
