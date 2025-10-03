# cluster.py – SBERT clustering using SentenceTransformers
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"   # compact + accurate

def cluster_profiles_with_sbert(csv_path: str = "data/sample_profiles.csv",
                                n_clusters: int = 10):
    # 1) Load data
    df = pd.read_csv(csv_path)
    if "text" not in df.columns:
        raise ValueError("CSV must have a 'text' column")
    if "id" not in df.columns:
        raise ValueError("CSV must have an 'id' column")

    # 2) Load model
    model = SentenceTransformer(MODEL_NAME)

    # 3) Encode bios
    texts = df["text"].tolist()
    embs = model.encode(texts, convert_to_numpy=True)

    # 4) Cluster (ensure float64 for sklearn KMeans)
    embs64 = np.asarray(embs, dtype=np.float64)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(embs64)
    df["cluster"] = kmeans.labels_

    return df, model, kmeans
