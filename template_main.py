from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
import numpy as np

from sklearn.decomposition import PCA
import umap

from sklearn.cluster import KMeans

def dim_red(mat, p, method):
    if method=='ACP':
      pca = PCA(n_components=p)
      mat = pca.fit_transform(mat)
      red_mat = mat[:,:p]

    elif method=='UMAP':
      reducer = umap.UMAP(n_components=p)
      red_mat = reducer.fit_transform(mat)

    else:
        raise Exception("Please select one of the three methods : APC, AFC, UMAP")

    return red_mat


def clust(mat, k):
  kmeans = KMeans(n_clusters=k, random_state=42)
  kmeans.fit(mat)
  return kmeans.labels_

ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]
k = len(set(labels))

# embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(corpus)

methods = ['ACP','UMAP']
for method in methods:
    # Perform dimensionality reduction
    red_emb = dim_red(embeddings, 20, method)

    # Perform clustering
    pred = clust(red_emb, k)

    # Evaluate clustering results
    nmi_score = normalized_mutual_info_score(pred, labels)
    ari_score = adjusted_rand_score(pred, labels)

    # Print results
    print(f'Method: {method}\nNMI: {nmi_score:.2f} \nARI: {ari_score:.2f}\n')