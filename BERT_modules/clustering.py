

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import torch
import pandas as pd


def compute_kmeans_clustering(relation_embeddings: pd.DataFrame, n_rel: int, \
    random_state: int):
    """Compute kmeans clustering with fixed nb of clusters
    Args:
        relation_embeddings (pd.DataFrame): relation embeddings
        n_rel (int): number of relations (nb of clusters)
    Returns:
        torch.Tensor: predicted labels
    """
    embeddings = relation_embeddings['mask_embedding'].tolist()

    model = KMeans(init='k-means++', n_init=10, n_clusters=n_rel, random_state=random_state, algorithm='elkan')
    predicted_labels = model.fit(embeddings)
    predicted_labels = model.predict(embeddings)
    
    return predicted_labels



def plot_elbow_curve(data: pd.DataFrame, max_k: int):
    """
    Plot the elbow curve for KMeans clustering using Seaborn.
    
    Args:
        data (numpy.ndarray or torch.Tensor): The data for clustering.
        max_k (int): The maximum number of clusters to consider.
    """
    wcss = []
    data = data['mask_embedding'].tolist()

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init= 'auto', algorithm='elkan')
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, max_k), y=wcss, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal k')
    plt.show();


