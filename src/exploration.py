# Brenden Collins // Nate Novak
# CS 7180: Advanced Computer Perception
# Fall 2022
# Perform dimension reduction and clustering analyses on encoded sentence data 


import numpy as np
import pandas as pd
import sklearn
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

def main(): 
    plt.rcParams['figure.figsize'] = [12, 8]
    # load data

    print("Loading data...")
    cls_out = pd.read_csv("../data/encoded.csv")

    cls_np = np.array(cls_out.drop(["Unnamed: 0", "PhraseId", "SentenceId", "Phrase"], axis=1))

    sentiment = cls_np[:,0]
    vector = cls_np[:,1:]

    # PCA on all sentences - CLS vector only
    pca = sklearn.decomposition.PCA()
    pca.fit(vector)
       
    print(f"Num components: {pca.n_components_}")
    print(f"First 30 axes variance: {pca.explained_variance_ratio_[:30]}")
    test = np.array(pca.explained_variance_ratio_)[:20].sum()
    print(f"Percent of variance in the first 20 axes: {test}")

    # apply PCA to transform the vector
    
    print("Running PCA...")
    pca_transform = sklearn.decomposition.PCA(n_components=10)
    vector_pca = pca_transform.fit_transform(vector)

    print(f"Num components: {pca_transform.n_components_}")
    print(f"First 10 axes variance: {pca_transform.explained_variance_ratio_[:10]}")
    test2 = np.array(pca_transform.explained_variance_ratio_)[:10].sum()
    print(f"Percent of variance in the first 10 axes: {test2}")

    print("Conducting UMAP analysis on all dimensions...")
    reducer0 = umap.UMAP()
    embedding0 = reducer0.fit_transform(vector_pca)

    # assign colors
    colors = sentiment.astype(int)

    fig, ax = plt.subplots(figsize=(12,10))

    plt.scatter(
        embedding0[:, 0],
        embedding0[:, 1],
        s=1,
        c=colors,
        )
    plt.gca().set_aspect('equal', 'datalim')
    plt.setp(ax, xticks=[], yticks=[])
    plt.title('UMAP projection of All Principal Components of the Encoded dataset', fontsize=18)

    plt.show()

    plt.savefig("../out/umap_image_alldims.png")

    clustering = SpectralClustering(n_clusters=5, assign_labels='discretize', random_state=0).fit(vector_pca)
    clustering

    # assign colors
    colors = sentiment.astype(int)

    fig, ax = plt.subplots(figsize=(12,10))

    plt.scatter(
        embedding0[:, 0],
        embedding0[:, 1],
        s=1,
        c=clustering.labels_,
        )
    plt.gca().set_aspect('equal', 'datalim')
    plt.setp(ax, xticks=[], yticks=[])
    plt.title('Spectral clustering of 100 Principal Components of the Encoded dataset', fontsize=18)

    plt.show()

    plt.savefig("../out/specClust.png")
    
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(vector)

    embedding.shape

    # assign colors
    colors = sentiment.astype(int)

    fig, ax = plt.subplots(figsize=(12,10))

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=1,
        c=colors,
        )

    plt.setp(ax, xticks=[], yticks=[])
    plt.title('UMAP projection of the Encoded dataset', fontsize=18)

    plt.show()


if __name__== "__main__": 
    main()
