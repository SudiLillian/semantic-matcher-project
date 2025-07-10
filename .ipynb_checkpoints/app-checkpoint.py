import streamlit as st
import pandas as pd
from matcher import Matcher
from explain import Explainer
import matplotlib.pyplot as plt

st.set_page_config(page_title='Semantic Profile Matcher', page_icon='🔍')
st.title('🔍 Semantic Profile Matcher')

uploaded = st.file_uploader('Upload CSV with columns id,text', type='csv')
if not uploaded:
    st.info('Using bundled sample_profiles.csv')
    uploaded = open('data/sample_profiles.csv', 'rb')

profiles_df = pd.read_csv(uploaded)
matcher = Matcher(profiles_df)
explainer = Explainer(matcher.vectorizer, matcher.tfidf_matrix)

query = st.text_input('Enter a search query')
if query:
    top_n = st.slider('How many results?', 3, 20, 5)
    results = matcher.match(query, top_n)
    st.subheader('Results')
    for idx, (rid, text, score) in enumerate(results):
        with st.expander(f'{idx+1}. **{rid}** — Score {score:.3f}'):
            st.write(text)
            if st.button('Explain ▶️', key=f'btn_{idx}'):
                doc_idx = matcher.ids.index(rid)
                tokens, shap_vals = explainer.explain(query, doc_idx)
                fig, ax = plt.subplots()
                colors = ['red' if v > 0 else 'blue' for v in shap_vals]
                ax.barh(tokens, shap_vals, color=colors)
                ax.set_xlabel('SHAP value (impact on similarity)')
                ax.invert_yaxis()
                st.pyplot(fig)
