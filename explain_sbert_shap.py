# explain_sbert_shap.py – True SHAP-based SBERT Explainer
import numpy as np
import shap
from sklearn.metrics.pairwise import cosine_similarity

class SBERTExplainer:
    """True SHAP explanation for SBERT similarity."""
    
    def __init__(self, sbert_model):
        self.model = sbert_model
    
    def subsift_tokenizer(self, s, separator=r"\W", return_offsets_mapping=True):
        import re
        pos = 0
        offset_ranges, input_ids = [], []
        for m in re.finditer(separator, s):
            start, end = m.span(0)
            offset_ranges.append((pos, start))
            input_ids.append(s[pos:start])
            pos = end
        if pos != len(s):
            offset_ranges.append((pos, len(s)))
            input_ids.append(s[pos:])
        out = {"input_ids": input_ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offset_ranges
        return out
    
    def explain(self, query, doc_text, top_k=15, max_evals=50):
        """Generate SHAP explanation for SBERT similarity"""
        
        def f(x, query_text=query, model=self.model):
            text_embeddings = model.encode(list(x), convert_to_numpy=True)
            query_embedding = model.encode([query_text], convert_to_numpy=True)
            return cosine_similarity(text_embeddings, query_embedding).flatten()
        
        masker = shap.maskers.Text(self.subsift_tokenizer)
        explainer = shap.Explainer(f, masker)
        shap_values = explainer([doc_text], max_evals=max_evals)
        html_shap = shap.plots.text(shap_values[0], display=False)
        
        try:
            tokens = shap_values[0].data
            contributions = shap_values[0].values
            valid_indices = [i for i, token in enumerate(tokens) if token.strip()]
            tokens = [tokens[i].strip() for i in valid_indices]
            contributions = [contributions[i] for i in valid_indices]
            return tokens, contributions, html_shap
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return [], [], None