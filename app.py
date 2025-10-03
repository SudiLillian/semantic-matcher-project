import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.parse
from sklearn.feature_extraction.text import CountVectorizer

from cluster import cluster_profiles_with_sbert
from matcher_transformers import rerank
from explain_sbert_shap import SBERTExplainer

# ------------------ Advanced TF-IDF Vectorizer ------------------
def create_advanced_tfidf_vectorizer(feature_number=3000):
    """Create TF-IDF vectorizer with single words, bi-grams and tri-grams for better accuracy"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    extended_stop_words = set([
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to',
        'was', 'will', 'with', 'would', 'have', 'had', 'this', 'these', 'they',
        'them', 'their', 'there', 'where', 'when', 'what', 'who', 'why', 'how',
        'can', 'could', 'should', 'may', 'might', 'must', 'shall',
        'years', 'including', 'such', 'also', 'well', 'within', 'across', 
        'through', 'during', 'over', 'under', 'between', 'among', 'around', 
        'about', 'above', 'below', 'into', 'onto',
        'i', 'you', 'we', 'us', 'our', 'my', 'your', 'his', 'her', 'him'
    ])
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        stop_words=list(extended_stop_words),
        max_features=feature_number,
        lowercase=True,
        token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b',
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        norm='l2'
    )
    return vectorizer

# ------------------ Token Filtering ------------------
def filter_meaningful_tokens(tokens, contributions, min_importance=0.01):
    if not tokens or not contributions:
        return [], []
    tokens = np.array(tokens)
    contributions = np.array(contributions, dtype=float)
    importance_mask = np.abs(contributions) >= min_importance
    if not importance_mask.any():
        top_indices = np.argsort(np.abs(contributions))[::-1][:10]
        return tokens[top_indices].tolist(), contributions[top_indices].tolist()
    filtered_tokens = tokens[importance_mask]
    filtered_contributions = contributions[importance_mask]
    generic_patterns = [
        'years experience', 'work experience', 'working with', 'experience with',
        'responsible for', 'worked on', 'involved in', 'participated in',
        'member of', 'part of', 'worked as', 'served as', 'acted as'
    ]
    semantic_tokens, semantic_contributions = [], []
    for token, contrib in zip(filtered_tokens, filtered_contributions):
        if any(pattern in token.lower() for pattern in generic_patterns):
            continue
        if len(token.split()) >= 2:
            semantic_tokens.append(token)
            semantic_contributions.append(contrib)
        elif len(token) > 3 and token.isalpha():
            semantic_tokens.append(token)
            semantic_contributions.append(contrib)
    if len(semantic_tokens) < 5:
        top_indices = np.argsort(np.abs(filtered_contributions))[::-1][:15]
        return filtered_tokens[top_indices].tolist(), filtered_contributions[top_indices].tolist()
    return semantic_tokens[:15], semantic_contributions[:15]

# ------------------ SubSift-style SHAP Helpers ------------------
def subsift_tokenizer(s, separator=r"\W", return_offsets_mapping=True):
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

def create_subsift_explainer(query, tfidf_vectorizer):
    def f(x, fixed_text=query, tfidf=tfidf_vectorizer):
        from sklearn.metrics.pairwise import cosine_similarity
        features = np.array(tfidf.transform(x).todense())
        match_features = np.array(tfidf.transform([fixed_text]).todense())
        return cosine_similarity(features, match_features).flatten()
    return f

def explain_with_subsift_shap(query, text_to_explain, tfidf_vectorizer, max_evals=100):
    import shap
    f = create_subsift_explainer(query, tfidf_vectorizer)
    masker = shap.maskers.Text(subsift_tokenizer)
    explainer = shap.Explainer(f, masker)
    shap_values = explainer([text_to_explain], max_evals=max_evals)
    html_shap = shap.plots.text(shap_values[0], display=False)
    try:
        tokens = shap_values[0].data
        contributions = shap_values[0].values
        valid_indices = [i for i, token in enumerate(tokens) if token.strip()]
        tokens = [tokens[i].strip() for i in valid_indices]
        contributions = [contributions[i] for i in valid_indices]
        return tokens, contributions, None, html_shap
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        return [], [], []

# ------------------ Helpers ------------------
def extract_keywords(texts, top_n=10):
    if not texts:
        return []
    vec = CountVectorizer(ngram_range=(1,2), stop_words="english").fit(texts)
    bag = vec.transform(texts)
    counts = bag.sum(axis=0).A1
    vocab = vec.get_feature_names_out()
    top_keywords = sorted(zip(vocab, counts), key=lambda x: -x[1])[:top_n]
    return [kw for kw, _ in top_keywords]

def build_linkedin_search(name: str) -> str:
    return "https://www.linkedin.com/search/results/all/?keywords=" + urllib.parse.quote_plus(name)

def star_rating(sim, worst=0.30, best=0.90):
    sim_clamped = max(worst, min(best, float(sim)))
    stars = int(round(1 + 4 * (sim_clamped - worst) / (best - worst)))
    return "★" * stars + "☆" * (5 - stars)

def plot_simple_top_tokens(tokens, contributions, top_n=5):
    """Display top N tokens as simple horizontal bars"""
    if len(contributions) == 0 or all(c == 0 for c in contributions):
        st.info("No meaningful contributions to display.")
        return
    
    contributions = np.array(contributions, dtype=float)
    sorted_indices = np.argsort(np.abs(contributions))[::-1][:top_n]
    top_tokens = [tokens[i] for i in sorted_indices]
    top_contribs = np.abs(contributions[sorted_indices])
    
    if len(top_tokens) == 0:
        st.info("No meaningful contributions to display.")
        return
    
    fig, ax = plt.subplots(figsize=(10, len(top_tokens) * 0.5))
    y_pos = np.arange(len(top_tokens))
    
    ax.barh(y_pos, top_contribs, color='steelblue', edgecolor='white', linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_tokens, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('Contribution Score', fontsize=12)
    ax.set_title('Top 5 Most Influential Tokens', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
def show_feedback_section():
    # Use mode-specific key to avoid duplicates
    mode_key = st.session_state.get('current_mode', 'default')
    query_hash = hash(st.session_state.get('current_query', ''))
    unique_key = f"{mode_key}_{query_hash}"
    
    st.markdown("---")
    with st.expander("💬 Was this helpful?", expanded=False):
        col1, col2 = st.columns([1, 3])
        with col1:
            helpful = st.radio("", ["👍 Yes", "👎 No"], key=f"feedback_{unique_key}")
        with col2:
            if helpful == "👎 No":
                feedback = st.text_area(
                    "Please tell us how we can improve:", 
                    placeholder="Your feedback helps us improve the search and explanation quality...",
                    key=f"feedback_text_{unique_key}"
                )
                if st.button("Submit Feedback", key=f"submit_feedback_{unique_key}"):
                    st.success("Thank you! Your feedback has been recorded.")
            else:
                st.success("Great! Thank you for your feedback.")

# ------------------ Session State ------------------
def init_session_state():
    if "tf_results" not in st.session_state:
        st.session_state["tf_results"] = []
    if "tf_explanations" not in st.session_state:
        st.session_state["tf_explanations"] = {}
    if "tf_explained_items" not in st.session_state:
        st.session_state["tf_explained_items"] = set()
    if "sbert_results" not in st.session_state:
        st.session_state["sbert_results"] = []
    if "sbert_explanations" not in st.session_state:
        st.session_state["sbert_explanations"] = {}
    if "sbert_explained_items" not in st.session_state:
        st.session_state["sbert_explained_items"] = set()

# ------------------ UI ------------------
st.set_page_config(page_title="LEAP Profiles", page_icon="🔍")
st.title("LEAP Profiles")
init_session_state()

mode = st.radio("Choose matching mode:", ["Explainable (TF-IDF)", "Semantic (SBERT + Cluster)"])

uploaded = st.file_uploader("Upload CSV with columns: id,text", type="csv")
if not uploaded:
    st.info("Using bundled sample_profiles.csv")
    uploaded = open("data/sample_profiles.csv", "rb")
profiles_df = pd.read_csv(uploaded)

# ------------------ TF-IDF MODE ------------------
if mode == "Explainable (TF-IDF)":
    st.header("TF-IDF Search")

    @st.cache_resource
    def init_tfidf_components():
        advanced_vectorizer = create_advanced_tfidf_vectorizer(feature_number=5000)
        profile_texts = profiles_df['text'].tolist()
        tfidf_matrix = advanced_vectorizer.fit_transform(profile_texts)
        class AdvancedMatcher:
            def __init__(self, df, vectorizer, matrix):
                self.df = df
                self.vectorizer = vectorizer
                self.tfidf_matrix = matrix
                self.ids = df['id'].tolist()
            def match(self, query, top_n=5):
                from sklearn.metrics.pairwise import cosine_similarity
                query_vec = self.vectorizer.transform([query])
                similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
                top_indices = similarities.argsort()[::-1][:top_n]
                results = []
                for idx in top_indices:
                    results.append((self.ids[idx], self.df.iloc[idx]['text'], similarities[idx]))
                return results
        return AdvancedMatcher(profiles_df, advanced_vectorizer, tfidf_matrix)

    matcher = init_tfidf_components()

    col1, col2, col3 = st.columns([5, 2, 1])
    with col1:
        query = st.text_input("Enter search query", key="tf_query")
    with col2:
        top_n = st.slider("Results", 3, 20, 5, key="tf_top_n")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing to align button
        search_button = st.button("Search", key="tf_search_btn", type="primary")

    if search_button and query.strip():
        with st.spinner("Searching..."):
            results = matcher.match(query, top_n)
            st.session_state["tf_results"] = results
            st.session_state["last_tf_query"] = query
            st.session_state['current_mode'] = 'tfidf'
            st.session_state["tf_explanations"] = {}
            st.session_state["tf_explained_items"] = set()
            st.rerun()
    elif not query.strip():
        if st.session_state.get("tf_results"):
            st.session_state["tf_results"] = []
            st.session_state["tf_explanations"] = {}
            st.session_state["tf_explained_items"] = set()
            st.rerun()

    if st.session_state["tf_results"]:
        st.subheader("Search Results")
        results = st.session_state["tf_results"]
        for idx, (rid, text, score) in enumerate(results):
            with st.expander(f"{idx+1}. {rid} (score {score:.3f})", expanded=False):
                # Show plain text
                st.write(text)

                # Explain Match button
                explain_key = f"tf_explain_{idx}"
                if st.button("Explain Match", key=explain_key):
                    st.session_state["tf_explained_items"].add(idx)
                    st.rerun()

                # Generate and display explanations
                if idx in st.session_state["tf_explained_items"]:
                    exp_key = f"tf_exp_{idx}"
                    exp_key_shap = f"tf_exp_shap_{idx}"
                    
                    if exp_key not in st.session_state["tf_explanations"]:
                        with st.spinner("Generating explanation..."):
                            try:
                                tokens, vals, _, html_shap = explain_with_subsift_shap(query, text, matcher.vectorizer, max_evals=50)
                                st.session_state[exp_key_shap] = html_shap
                                
                                if not tokens or not vals:
                                    doc_idx = matcher.ids.index(rid)
                                    doc_vec = matcher.tfidf_matrix[doc_idx]
                                    if hasattr(doc_vec, 'toarray'):
                                        doc_vec = doc_vec.toarray()[0]
                                    query_vec = matcher.vectorizer.transform([query]).toarray()[0]
                                    if len(query_vec) != len(doc_vec):
                                        st.error(f"Dimension mismatch: query={len(query_vec)}, doc={len(doc_vec)}")
                                    else:
                                        contributions = query_vec * doc_vec
                                        feature_names = matcher.vectorizer.get_feature_names_out()
                                        mask = contributions != 0
                                        if mask.any():
                                            filtered_contributions = contributions[mask]
                                            filtered_tokens = feature_names[mask]
                                            tokens, vals = filter_meaningful_tokens(
                                                filtered_tokens.tolist(),
                                                filtered_contributions.tolist()
                                            )
                                        else:
                                            tokens, vals = [], []
                                else:
                                    tokens, vals = filter_meaningful_tokens(tokens, vals)
                                
                                st.session_state["tf_explanations"][exp_key] = (tokens, vals)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error computing explanation: {e}")
                                st.session_state["tf_explanations"][exp_key] = ([], [])

                    # Display SHAP
                    if exp_key_shap in st.session_state:
                        st.markdown("**🔬 SHAP Text Explanation:**")
                        st.html(st.session_state[exp_key_shap])
                        st.markdown("---")
                    
                    # Display TF-IDF bar chart
                    if exp_key in st.session_state["tf_explanations"]:
                        tokens, vals = st.session_state["tf_explanations"][exp_key]
                        if tokens and vals:
                            st.markdown("**🔍 TF-IDF Match Explanation:**")
                            plot_simple_top_tokens(tokens, vals, top_n=5)
                        else:
                            st.info("No meaningful word contributions found for this match.")
        
        show_feedback_section()

# ------------------ SBERT MODE ------------------
else:
    st.header("Semantic Search (SBERT + Clustering)")

    @st.cache_resource
    def init_sbert_pipeline():
        return cluster_profiles_with_sbert(csv_path="data/sample_profiles.csv", n_clusters=3)

    try:
        clustered_df, sbert_model, kmeans_model = init_sbert_pipeline()
        explainer_sbert = SBERTExplainer(sbert_model)
    except Exception as e:
        st.error(f"Error initializing SBERT components: {e}")
        st.stop()

    cluster_list = sorted(clustered_df["cluster"].unique())

    if "current_cluster" in st.session_state:
        st.sidebar.markdown("---")
        st.sidebar.subheader(f"🎯 Predicted Cluster: {st.session_state['current_cluster']}")
        cluster_members = clustered_df[clustered_df["cluster"] == st.session_state["current_cluster"]]["id"].tolist()
        st.sidebar.markdown(f"**{len(cluster_members)} people in this cluster:**")
        with st.sidebar.container():
            for member in cluster_members[:20]:
                st.sidebar.write(f"• {member}")
            if len(cluster_members) > 20:
                st.sidebar.write(f"... and {len(cluster_members) - 20} more")
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Cluster Statistics:**")
        st.sidebar.write(f"Total profiles: {len(cluster_members)}")
        if "available_tags" in st.session_state:
            st.sidebar.markdown("**Common terms:**")
            for tag in st.session_state["available_tags"][:5]:
                st.sidebar.write(f"• {tag}")
    elif st.session_state.get("sbert_results"):
        st.sidebar.info("👆 Perform a search to see cluster analysis")

    if "sbert_results" in st.session_state and st.session_state["sbert_results"]:
        st.info(f"Found {len(st.session_state['sbert_results'])} matches in cluster {st.session_state.get('current_cluster', 'N/A')}")
        if st.button("📊 Export Results", key="export_results"):
            results_data = []
            for idx, (rid, text, sim_score, cluster_label) in enumerate(st.session_state["sbert_results"]):
                results_data.append({
                    "Rank": idx + 1,
                    "ID": rid,
                    "Text": text,
                    "Similarity": sim_score,
                    "Cluster": cluster_label
                })
            results_df = pd.DataFrame(results_data)
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"search_results_{st.session_state.get('current_query', 'query')}.csv",
                mime="text/csv"
            )

    col1, col2, col3 = st.columns([5, 2, 1])
    with col1:
        query = st.text_input("Enter search query", key="sbert_query")
    with col2:
        top_n = st.slider("Results", 3, 20, 5, key="sbert_top_n")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing to align button
        search_button = st.button("Search", key="sbert_search_btn", type="primary")

    if search_button and query.strip():
        with st.spinner("Searching and clustering..."):
            try:
                q_vec = np.array(sbert_model.encode([query], convert_to_numpy=True), dtype=np.float64)
                cluster_id = kmeans_model.predict(q_vec)[0]
                cand_df = clustered_df[clustered_df["cluster"] == cluster_id].copy()
                tags = extract_keywords(cand_df["text"].tolist(), top_n=20)
                st.session_state["available_tags"] = tags
                st.session_state["candidate_df"] = cand_df
                st.session_state["current_query"] = query
                st.session_state['current_mode'] = 'sbert'
                st.session_state["current_cluster"] = cluster_id
                st.session_state["last_sbert_query"] = query
                results = rerank(query, cand_df, top_k=top_n, model=sbert_model)
                st.session_state["sbert_results"] = results
                st.session_state["sbert_explanations"] = {}
                st.session_state["sbert_explained_items"] = set()
                st.rerun()
            except Exception as e:
                st.error(f"Error during SBERT search: {e}")
    elif not query.strip():
        if st.session_state.get("sbert_results"):
            st.session_state["sbert_results"] = []
            st.session_state["sbert_explanations"] = {}
            st.session_state["sbert_explained_items"] = set()
            st.rerun()

    if "available_tags" in st.session_state:
        sel_tags = st.sidebar.multiselect("Refine by tags", st.session_state["available_tags"])
        if st.sidebar.button("🔄 Apply Filters", key="apply_filters") or sel_tags != st.session_state.get("last_selected_tags", []):
            with st.spinner("Applying filters and reranking..."):
                try:
                    cand_df = st.session_state["candidate_df"].copy()
                    if sel_tags:
                        pattern = "|".join(sel_tags)
                        cand_df = cand_df[cand_df["text"].str.contains(pattern, case=False, regex=True)]
                    if not cand_df.empty:
                        results = rerank(st.session_state["current_query"], cand_df,
                                         top_k=st.session_state.get("sbert_top_n", 5), model=sbert_model)
                        st.session_state["sbert_results"] = results
                        st.session_state["last_selected_tags"] = sel_tags
                        st.session_state["sbert_explanations"] = {}
                        st.session_state["sbert_explained_items"] = set()
                        st.rerun()
                    else:
                        st.warning("No profiles match the selected tags.")
                        st.session_state["sbert_results"] = []
                except Exception as e:
                    st.error(f"Error during reranking: {e}")

    if st.session_state["sbert_results"]:
        st.subheader("Semantic Search Results")
        results = st.session_state["sbert_results"]
        for idx, (rid, text, sim_score, cluster_label) in enumerate(results):
            # ---------- FIX: define explanation_key up-front ----------
            explanation_key = f"sbert_exp_{idx}"
            # ---------------------------------------------------------
            rating = star_rating(sim_score)
            with st.expander(f"{idx+1}. {rid} [Cluster {cluster_label}] {rating}", expanded=False):
                
                # Show plain text
                st.write(text)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"[🔗 View on LinkedIn]({build_linkedin_search(rid)})")
                with col2:
                    st.markdown(f"**Similarity:** {sim_score:.3f}")

                explain_btn_key = f"sbert_explain_btn_{idx}"
                if st.button("Explain Match", key=explain_btn_key):
                    st.session_state["sbert_explained_items"].add(idx)
                    st.rerun()

                if idx in st.session_state["sbert_explained_items"]:
                    if explanation_key not in st.session_state["sbert_explanations"]:
                        with st.spinner("Generating explanation..."):
                            try:
                                result = explainer_sbert.explain(st.session_state["current_query"], text, top_k=15)
                                if len(result) == 3:
                                    tokens, vals, html_shap = result
                                    st.session_state[f"sbert_exp_shap_{idx}"] = html_shap
                                elif len(result) == 2:
                                    tokens, vals = result
                                else:
                                    tokens, vals = [], []
                                
                                if tokens and vals:
                                    tokens, vals = filter_meaningful_tokens(tokens, vals)
                                else:
                                    tokens, vals = [], []
                                
                                st.session_state["sbert_explanations"][explanation_key] = (tokens, vals)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error computing explanation: {e}")
                                st.session_state["sbert_explanations"][explanation_key] = ([], [])
                    
                    # Display explanation (parallel to generation block)
                    exp_key_shap = f"sbert_exp_shap_{idx}"
                    if exp_key_shap in st.session_state and st.session_state[exp_key_shap] is not None:
                        st.markdown("**🔬 SHAP Text Explanation:**")
                        st.html(st.session_state[exp_key_shap])
                        st.markdown("---")
                    
                    # Display bar chart
                    if explanation_key in st.session_state["sbert_explanations"]:
                        tokens, vals = st.session_state["sbert_explanations"][explanation_key]
                        if tokens and vals:
                            st.markdown("**🔍 Match Explanation:**")
                            plot_simple_top_tokens(tokens, vals, top_n=5)
                        else:
                            st.info("No meaningful word contributions found for this match.")
        show_feedback_section()

