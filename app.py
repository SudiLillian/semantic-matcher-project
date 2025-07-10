import streamlit as st
import pandas as pd
from matcher import Matcher
# from explain import Explainer
from explain_fast import Explainer
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

    		# ----- get top-15 token contributions -----
    		tokens, vals = explainer.explain(query, doc_idx, top_k=15)

    		# ----- normalise so longest bar = 1 -----
    		if vals.max() != 0:
        		vals = vals / vals.max()

    		# ----- plotting -----
    		fig, ax = plt.subplots(figsize=(6, 4))
    		bar_colors = ['steelblue'] * len(vals)

    		# height=0.4 → thinner bars
    		ax.barh(tokens, vals, height=0.25, color=bar_colors, edgecolor='black')

    		ax.set_xlabel('Normalised contribution')
    		ax.invert_yaxis()             # highest bar at top
    		ax.set_xlim(0, 1)             # bars span 0-1
    		st.pyplot(fig)

