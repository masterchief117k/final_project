# =============================================================================
# 0) Install dependencies if you haven’t already:
#    pip install nltk scikit-learn matplotlib
# =============================================================================

import re
import nltk
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets            import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition       import TruncatedSVD, PCA
from sklearn.cluster             import KMeans
from sklearn.metrics             import silhouette_score
from sklearn.pipeline            import Pipeline
from sklearn.preprocessing       import Normalizer
from sklearn.model_selection     import ParameterGrid

from nltk.corpus       import stopwords
from nltk.stem         import WordNetLemmatizer
from nltk.tokenize     import RegexpTokenizer

# 1) Download NLTK data once
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# =============================================================================
# 2) Load & Clean Text
# =============================================================================
data     = fetch_20newsgroups(subset='all',
                              remove=('headers','footers','quotes'),
                              shuffle=True, random_state=42)
raw_docs = data.data

tokenizer  = RegexpTokenizer(r"[A-Za-z']+")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # strip quotes and collapse newlines
    text = re.sub(r">.+", " ", text)
    text = re.sub(r"\n+", " ", text)
    # tokenize, lowercase, drop stopwords, lemmatize
    toks = tokenizer.tokenize(text.lower())
    return " ".join(
        lemmatizer.lemmatize(t) for t in toks
        if t not in stop_words
    )

cleaned = [clean_text(doc) for doc in raw_docs]
# drop any empty docs
cleaned = [doc for doc in cleaned if doc.strip()]
print(f"→ {len(cleaned)} non-empty documents after cleaning")

# quick sanity check of vocab size
toy_vocab = CountVectorizer().fit(cleaned).vocabulary_
print(f"→ initial vocab size: {len(toy_vocab)} tokens")

# =============================================================================
# 3) Build the base pipeline (with placeholder hyperparams)
# =============================================================================
pipe = Pipeline([
    ('tfidf',   TfidfVectorizer()),
    ('norm',    Normalizer()),         # unit-length = cosine similarity
    ('svd',     TruncatedSVD(random_state=42)),
    ('cluster', KMeans(random_state=42))
])

# =============================================================================
# 4) Define your hyperparameter grid
# =============================================================================
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
# 5) Manual grid search with try/except
# =============================================================================
best_score, best_params = -1, None

for params in ParameterGrid(param_grid):
    try:
        # set pipeline hyperparameters
        pipe.set_params(**params)
        # fit transforms all steps
        pipe.fit(cleaned)

        # extract reduced data for silhouette
        X_tfidf  = pipe.named_steps['tfidf'].transform(cleaned)
        X_norm   = pipe.named_steps['norm'].transform(X_tfidf)
        X_reduced= pipe.named_steps['svd'].transform(X_norm)
        labels   = pipe.named_steps['cluster'].labels_

        score = silhouette_score(X_reduced, labels, metric='euclidean')

        if score > best_score:
            best_score, best_params = score, params
            print(f"New best {best_score:.3f} with {best_params}")

    except ValueError:
        # skip combos that produce empty vocab or other errors
        continue

print("\n=== Grid search complete ===")
print(f"Best silhouette score: {best_score:.3f}")
print(f"Best parameters:       {best_params}")

# =============================================================================
# 6) Fit final pipeline on all data
# =============================================================================
pipe.set_params(**best_params)
pipe.fit(cleaned)

# 7) 2D Visualization via PCA
X_tfidf   = pipe.named_steps['tfidf'].transform(cleaned)
X_norm    = pipe.named_steps['norm'].transform(X_tfidf)
X_reduced = pipe.named_steps['svd'].transform(X_norm)
labels    = pipe.named_steps['cluster'].labels_

coords = PCA(n_components=2, random_state=42).fit_transform(X_reduced)

plt.figure(figsize=(8,6))
plt.scatter(coords[:,0], coords[:,1],
            c=labels, cmap='tab20', s=4)
plt.title("Final K-Means Clusters (2D PCA Projection)")
plt.xticks([]); plt.yticks([])
plt.show()
# assume you ran:
#   grid.fit(cleaned)
# and then:

# 1) Inject the winning params into your pipeline
pipe.set_params(**best_params)

# 2) Fit it on the entire cleaned corpus
pipe.fit(cleaned)

# 3) Alias it as your best estimator
best_pipe = pipe

# pull out the TF-IDF vectorizer and the final KMeans
tfidf = best_pipe.named_steps['tfidf']
km    = best_pipe.named_steps['cluster']

# now this is safe, because tfidf has been fit
features = tfidf.get_feature_names_out()
centers  = km.cluster_centers_
tfidf = best_pipe.named_steps['tfidf']
km    = best_pipe.named_steps['cluster']

# Get feature names and cluster centers
features = tfidf.get_feature_names_out()
centers  = km.cluster_centers_

# Print top terms per cluster
for cid, centroid in enumerate(centers):
    top_idxs  = centroid.argsort()[-10:][::-1]
    top_terms = [features[i] for i in top_idxs]
    print(f"Cluster {cid}: {', '.join(top_terms)}")

#%%
labels = km.labels_
for cid in range(km.n_clusters):
    print(f"\n=== Cluster {cid} Samples ===")
    samples = [doc for doc, lab in zip(cleaned, labels) if lab == cid][:3]
    for s in samples:
        print(" •", s[:200].replace("\n"," "), "…")
import streamlit as st

# 1) pull fitted pieces from your best pipeline
tfidf = best_pipe.named_steps['tfidf']
km    = best_pipe.named_steps['cluster']

# 2) compute top terms per cluster
features = tfidf.get_feature_names_out()
centers  = km.cluster_centers_
cluster_top_terms = {}
for cid, centroid in enumerate(centers):
    top10_idxs = centroid.argsort()[-10:][::-1]
    cluster_top_terms[cid] = [features[i] for i in top10_idxs]

# 3) gather example docs per cluster
labels = km.labels_
examples_by_cluster = {cid: [] for cid in range(km.n_clusters)}
for doc, lab in zip(cleaned, labels):
    examples_by_cluster[lab].append(doc)

# 4) Streamlit UI
st.title("20 Newsgroups: K-Means Cluster Explorer")

# Sidebar: pick a cluster
cluster_choice = st.sidebar.selectbox(
    label="Select cluster",
    options=list(range(km.n_clusters)),
    format_func=lambda x: f"Cluster {x}"
)

# Show top terms
st.subheader(f"Top terms for cluster {cluster_choice}")
st.write(", ".join(cluster_top_terms[cluster_choice]))

# Show a few sample documents
st.subheader(f"Sample documents for cluster {cluster_choice}")
for ex in examples_by_cluster[cluster_choice][:5]:
    st.markdown(f"> {ex[:300].replace(chr(10), ' ')} …")
