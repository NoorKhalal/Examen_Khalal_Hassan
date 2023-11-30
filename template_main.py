import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import umap
from sentence_transformers import SentenceTransformer
import argparse



def dim_red(mat, p, method):
    if method=='ACP':
      pca = PCA(n_components=p)
      mat = pca.fit_transform(mat)
      red_mat = mat[:,:p]

    elif method=='UMAP':
      reducer = umap.UMAP(n_components=p)
      red_mat = reducer.fit_transform(mat)
    
    elif method=='T-SNE':
        tsne = TSNE(n_components=p, init='pca', perplexity=50, random_state=42, n_jobs=-1) #init peut etre 'random' aussi
        red_mat = tsne.fit_transform(mat)
      
    else:
        raise Exception("Please select one of the three methods : T-SNE, AFC, UMAP")

    return red_mat


def clust(mat, k):
  kmeans = KMeans(n_clusters=k, random_state=42)
  kmeans.fit(mat)
  return kmeans.labels_


# methods that are available in this script
methods = ['ACP', 'UMAP', 'T-SNE']

# Create an argument parser
parser = argparse.ArgumentParser(description='Dimensionality Reduction and Clustering')

# Add an argument for choosing the method
parser.add_argument('--method', choices=methods, help='Choose a dimensionality reduction method')

# Parse the arguments
args = parser.parse_args()

# Use the chosen method or default to 'ACP' if no method is provided
chosen_method = args.method if args.method else 'ACP'

nb_dim = 3 if chosen_method == 'T-SNE' else 20

ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]
k = len(set(labels))

# embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(corpus)


# Rest of your code
for method in methods:
    if method == chosen_method:
        red_emb = dim_red(embeddings, nb_dim, method)
    else:
        continue

    pred = clust(red_emb, k)

    nmi_score = normalized_mutual_info_score(pred, labels)
    ari_score = adjusted_rand_score(pred, labels)

    print(f'Method: {method}\nNMI: {nmi_score:.2f} \nARI: {ari_score:.2f}\n')
