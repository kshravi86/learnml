# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Define a function to visualize GMM clustering results
def visualize_gmm_clustering(X, gmm_labels, title):
    plt.figure(figsize=(5, 5))
    plt.scatter(X[:, 0], X[:, 1], c=gmm_labels)
    plt.title(title)
    plt.show()

# Limitation 1: GMMs assume spherical clusters
X = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [3, 3], [3, 4], [4, 3], [4, 4]])
gmm = GaussianMixture(n_components=2)
gmm_labels = gmm.fit_predict(X)
visualize_gmm_clustering(X, gmm_labels, "GMM Clustering with Spherical Clusters")

# Limitation 2: GMMs are sensitive to the initial placement of centroids
gmm_init_bad = GaussianMixture(n_components=2, init_params='random')
gmm_init_bad_labels = gmm_init_bad.fit_predict(X)
visualize_gmm_clustering(X, gmm_init_bad_labels, "GMM Clustering with Bad Initialization")

# Limitation 3: GMMs can be computationally expensive for high-dimensional data
X_high_dim = np.random.rand(1000, 100)  # Generate high-dimensional data
gmm_high_dim = GaussianMixture(n_components=5)
gmm_high_dim_labels = gmm_high_dim.fit_predict(X_high_dim)
visualize_gmm_clustering(X_high_dim, gmm_high_dim_labels, "GMM Clustering with High-Dimensional Data")
