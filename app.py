# =============================================================================
# 20 Newsgroups: K-Means Cluster Explorer (Streamlit)
#
#   pip install -r requirements.txt
#   streamlit run app.py
#
# The first launch downloads the dataset and fits the model (a minute or two);
# after that, results are cached and the app responds instantly.
# =============================================================================

import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets            import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition       import TruncatedSVD, PCA
from sklearn.cluster             import KMeans
from sklearn.metrics             import silhouette_score
from sklearn.pipeline            import Pipeline
from sklearn.preprocessing       import Normalizer
from sklearn.model_selection     import ParameterGrid

from nltk.corpus       import stopwords
from nltk.stem         import WordNetLemmatizer
from nltk.tokenize     import RegexpTokenizer

# Best configuration found by the grid search below (silhouette = 0.135).
BEST_PARAMS = {
    'tfidf__ngram_range': (1, 2),
    'tfidf__min_df':       1,
    'tfidf__max_df':       0.8,
    'svd__n_components':   50,
    'cluster__n_clusters': 10,
    'cluster__init':       'k-means++',
    'cluster__n_init':     5,
}

# Set to True to re-run the full grid search (192 configurations; slow).
RUN_GRID_SEARCH = False

param_grid = {
    'tfidf__ngram_range': [(1,1), (1,2)],
    'tfidf__min_df':       [1, 5],
    'tfidf__max_df':       [0.8, 0.9],
    'svd__n_components':   [50, 100],
    'cluster__n_clusters': [10, 15, 20],
    'cluster__init':       ['k-means++', 'random'],
    'cluster__n_init':     [5, 10]
}


# =============================================================================
# 1) Load & clean text
# =============================================================================
tokenizer = RegexpTokenizer(r"[A-Za-z']+")

def clean_text(text, stop_words, lemmatizer):
    # strip quoted replies and collapse newlines
    text = re.sub(r">.+", " ", text)
    text = re.sub(r"\n+", " ", text)
    # tokenize, lowercase, drop stopwords, lemmatize
    toks = tokenizer.tokenize(text.lower())
    return " ".join(lemmatizer.lemmatize(t) for t in toks if t not in stop_words)

@st.cache_data(show_spinner="Downloading and cleaning 20 Newsgroups…")
def load_corpus():
    for pkg in ('stopwords', 'wordnet', 'omw-1.4'):
        nltk.download(pkg, quiet=True)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'),
                              shuffle=True, random_state=42)
    cleaned = [clean_text(doc, stop_words, lemmatizer) for doc in data.data]
    return [doc for doc in cleaned if doc.strip()]   # drop empty docs


# =============================================================================
# 2) Pipeline: TF-IDF -> unit-normalise -> LSA (SVD) -> K-Means
# =============================================================================
def build_pipeline():
    return Pipeline([
        ('tfidf',   TfidfVectorizer()),
        ('norm',    Normalizer()),         # unit-length = cosine similarity
        ('svd',     TruncatedSVD(random_state=42)),
        ('cluster', KMeans(random_state=42))
    ])

def reduced_features(pipe, docs):
    X = pipe.named_steps['tfidf'].transform(docs)
    X = pipe.named_steps['norm'].transform(X)
    return pipe.named_steps['svd'].transform(X)

def grid_search(docs):
    best_score, best_params = -1, None
    for params in ParameterGrid(param_grid):
        try:
            pipe = build_pipeline().set_params(**params).fit(docs)
            score = silhouette_score(reduced_features(pipe, docs),
                                     pipe.named_steps['cluster'].labels_)
            if score > best_score:
                best_score, best_params = score, params
                print(f"New best {best_score:.3f} with {best_params}")
        except ValueError:
            continue   # skip combos that produce an empty vocabulary
    return best_params

@st.cache_resource(show_spinner="Fitting TF-IDF, LSA and K-Means…")
def fit_model(docs):
    params = grid_search(docs) if RUN_GRID_SEARCH else BEST_PARAMS
    pipe = build_pipeline().set_params(**params).fit(docs)
    X_reduced = reduced_features(pipe, docs)
    labels = pipe.named_steps['cluster'].labels_
    score = silhouette_score(X_reduced, labels, sample_size=5000, random_state=42)

    # Top terms: cluster centres live in the 50-d LSA space, so map them back
    # to the TF-IDF vocabulary before ranking words.
    centers_tfidf = pipe.named_steps['svd'].inverse_transform(
        pipe.named_steps['cluster'].cluster_centers_)
    vocab = pipe.named_steps['tfidf'].get_feature_names_out()
    top_terms = {cid: [vocab[i] for i in centre.argsort()[-10:][::-1]]
                 for cid, centre in enumerate(centers_tfidf)}

    coords = PCA(n_components=2, random_state=42).fit_transform(X_reduced)
    return params, labels, top_terms, coords, score


# =============================================================================
# 3) Streamlit UI
# =============================================================================
st.title("20 Newsgroups: K-Means Cluster Explorer")

cleaned = load_corpus()
params, labels, top_terms, coords, score = fit_model(cleaned)
n_clusters = params['cluster__n_clusters']
sizes = np.bincount(labels, minlength=n_clusters)

col1, col2, col3 = st.columns(3)
col1.metric("Documents", f"{len(cleaned):,}")
col2.metric("Clusters", n_clusters)
col3.metric("Silhouette (5k sample)", f"{score:.3f}")

cluster_choice = st.sidebar.selectbox(
    label="Select cluster",
    options=list(range(n_clusters)),
    format_func=lambda c: f"Cluster {c} · {', '.join(top_terms[c][:3])} ({sizes[c]:,})"
)

st.subheader(f"Top terms for cluster {cluster_choice}")
st.write(", ".join(top_terms[cluster_choice]))

st.subheader("Where this cluster sits (2-D PCA projection)")
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(coords[:, 0], coords[:, 1], c='lightgrey', s=3)
mask = labels == cluster_choice
ax.scatter(coords[mask, 0], coords[mask, 1], c='tab:blue', s=4)
ax.set_xticks([]); ax.set_yticks([])
st.pyplot(fig)

st.subheader(f"Sample documents ({sizes[cluster_choice]:,} in this cluster)")
examples = [doc for doc, lab in zip(cleaned, labels) if lab == cluster_choice]
for ex in examples[:5]:
    st.markdown(f"> {ex[:300]} …")
