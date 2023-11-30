# Data Engineer Exam: Dimensionality reduction and classification of the dataset 

this repository contains the python script and a jupyter notebook that applies a dimensionality reduction  at the dataset NG20. 

This Python script performs clustering and dimensionality reduction analysis on a dataset using various techniques, including KMeans, Agglomerative Clustering, Principal Component Analysis (PCA), Uniform Manifold Approximation and Projection (UMAP), and t-Distributed Stochastic Neighbor Embedding (T-SNE).


Multiple clustering methods and dimensionality reduction methods can be tested in our code such as Principal Component Analysis (PCA), Uniform Manifold Approximation and Projection (UMAP), and t-Distributed Stochastic Neighbor Embedding (T-SNE) for dimensionality reduction, and Kmeans, CAH with three linkage (ward, single, complete). 

The clustering methods can be ran multiple times too to showcase te mean and standard deviation of the NMI and ARI.

The evaluation of the methods can be seen in the NMI and ARI results, as well as in the plots in the notebook.

## Usage

To start off, it is better to create a venv

```bash
pip -m venv venv
source venv/bin/activate
```

then install the requirements:
```bash
pip install -r requirements.txt
```


In the python file, arguments can be added while running the file. A guide to all the arguments is shown below:

```bash
usage: template_main.py [-h]
                        [--dim_red_method {ACP,UMAP,T-SNE}]
                        [--clus_method {kmeans,CAH-ward,CAH-min,CAH-max}]
                        [--multipleruns MULTIPLERUNS]

Dimensionality Reduction and Clustering

options:
  -h, --help            show this help message and exit
  --dim_red_method {ACP,UMAP,T-SNE}
                        Choose a dimensionality reduction
                        method
  --clus_method {kmeans,CAH-ward,CAH-min,CAH-max}
                        Choose a clustering reduction
                        method
  --multipleruns MULTIPLERUNS
                        Number of runs for clustering
```

An example of running the file (in ubuntu):

```bash
python3 template_main.py --dim_red_method T-SNE --multipleruns 10
```

this line will apply the T-SNE dimensionality reduction on the data, then run the clustering algorithm 10 times. 
The default clustering algorithm is `Kmean`, `PCA` for dimensionality reduction, and 1 run for multiple runs. 


## Visualization
The notebook generates scatter plots using Plotly Express and Seaborn to visualize the clustering results in 2D and 3D.


## Data Loading
In an effort to optimize both execution time and the overall size of the Docker image, we have taken the initiative to store the embeddings and labels from the dataset. This decision has been implemented by saving the relevant information in two designated files: `embeddings.csv` and `labels.npy`. This approach not only enhances the efficiency of execution but also contributes to a more streamlined and resource-efficient Docker image, ensuring a smoother and more responsive deployment process.

## Link to the Dockerhub image:
https://hub.docker.com/layers/youssefhassan12/examen/latest/images/sha256:fcb1db5178a60836411e572621ad609c21716b3bf292e6d6968630f2fc343a73
