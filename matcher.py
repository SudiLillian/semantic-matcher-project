import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import text  # <- add

class Matcher:
    def __init__(self, profiles_df: pd.DataFrame):
        self.ids = profiles_df['id'].tolist()
        self.texts = profiles_df['text'].tolist()
    my_stop_words = text.ENGLISH_STOP_WORDS.union(
            {"ai", "and", "the", "research", "researcher"} 
    )
        self.vectorizer = TfidfVectorizer(
            stop_words=my_stop_words,
            ngram_range=(1, 2),
            max_features=5000,
        min_df=1
        sublinear_tf=True,                 # optional TF scaling
            analyzer="word",
            token_pattern=r"(?u)\b\w[\w+-]*\b"
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)


    def match(self, query: str, top_n: int = 5):
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        ranked = sorted(zip(self.ids, self.texts, scores),
                        key=lambda x: x[2], reverse=True)
        return ranked[:top_n]
