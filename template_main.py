import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering

import pandas as pd
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
        tsne = TSNE(n_components=p, init='pca', perplexity=50, n_jobs=-1) 
        red_mat = tsne.fit_transform(mat)
    else:
        raise Exception("Please select one of the three methods : APC, AFC, UMAP")

    return red_mat


def clus(mat, clus_tech ,k):
  if clus_tech=='kmeans':
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(mat)
    labels=kmeans.labels_
  elif clus_tech=='CAH-ward':
    agglomerative_clustering = AgglomerativeClustering(n_clusters=k)
    labels = agglomerative_clustering.fit_predict(mat)
  elif clus_tech=='CAH-min':
    agglomerative_clustering = AgglomerativeClustering(n_clusters=k,linkage='single')
    labels = agglomerative_clustering.fit_predict(mat)
  elif clus_tech=='CAH-max':
    agglomerative_clustering = AgglomerativeClustering(n_clusters=k,linkage='complete')
    labels = agglomerative_clustering.fit_predict(mat)

    
  return labels

# methods that are available in this script
methods = ['ACP', 'UMAP', 'T-SNE']
clustering_methods=['kmeans','CAH-ward','CAH-min','CAH-max']

# Create an argument parser
parser = argparse.ArgumentParser(description='Dimensionality Reduction and Clustering')

# Add an argument for choosing the method
parser.add_argument('--dim_red_method', choices=methods, default='ACP', help='Choose a dimensionality reduction method')

# Add an argument for choosing the method
parser.add_argument('--clus_method', choices=clustering_methods, default='kmeans', help='Choose a clustering reduction method')

# Add an argument for the number of runs
parser.add_argument('--multipleruns', type=int, default=1, help='Number of runs for clustering')

# Parse the arguments
args = parser.parse_args()

nb_dim = 3 if args.dim_red_method == 'T-SNE' else 20

# Perform multiple clustering calls


labels = np.load('labels.npy')
k=len(set(labels))

embeddings=pd.read_csv('embeddings.csv').values

red_emb = dim_red(embeddings, nb_dim, args.dim_red_method)


# Perform multiple clustering calls
nmi_scores = []
ari_scores = []


for _ in range(args.multipleruns):

    pred = clus(red_emb,args.clus_method, k)

    nmi_score = normalized_mutual_info_score(pred, labels)
    ari_score = adjusted_rand_score(pred, labels)

    nmi_scores.append(nmi_score)
    ari_scores.append(ari_score)

# Calculate mean and standard deviation
mean_nmi = np.mean(nmi_scores)
std_nmi = np.std(nmi_scores)

mean_ari = np.mean(ari_scores)
std_ari = np.std(ari_scores)

print(f'Results for {args.clus_method} on data reduced by {args.dim_red_method} with {args.multipleruns} iterations: ')
# Print the results
print(f'NMI: {mean_nmi:.2f} ± {std_nmi:.2f}')
print(f'ARI: {mean_ari:.2f} ± {std_ari:.2f}')