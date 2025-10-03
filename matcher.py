import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import text

class Matcher:
    """
    TF-IDF based profile matcher.
    Takes a DataFrame with 'id' and 'text' columns and returns
    the top-N most similar entries given a search query.
    """
    def __init__(self, profiles_df: pd.DataFrame):
        # Store profile IDs and their text descriptions
        self.ids   = profiles_df["id"].tolist()
        self.texts = profiles_df["text"].tolist()

        # Custom stopwords (extend built-in English stopwords)
        custom_stop_words = text.ENGLISH_STOP_WORDS.union({
            "and", "the", "is"
        })

        # Initialize and fit the TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words=list(custom_stop_words),
            ngram_range=(1, 2),              # Use unigrams and bigrams
            max_features=5000,
            min_df=1,
            sublinear_tf=True,
            analyzer="word",
            token_pattern=r"(?u)\b\w[\w+-]*\b",  # allow hyphenated tokens
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    # ─────────────────────────────────────────────────────
    def match(self, query: str, top_n: int = 5):
        """
        Find top-N most similar profiles using cosine similarity
        on TF-IDF vectors.

        Returns:
            List of tuples: (id, text, similarity_score)
        """
        # Vectorize the query using the same TF-IDF model
        q_vec = self.vectorizer.transform([query])

        # Compute cosine similarity between query and all profiles
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        # Sort and select top-N results
        ranked = sorted(
            zip(self.ids, self.texts, scores),
            key=lambda x: x[2],
            reverse=True
        )
        return ranked[:top_n]
