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
print("\nDemonstrating Limitation 1: Spherical Clusters Assumption")
print("Creating a simple dataset with two distinct clusters...")
X = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [3, 3], [3, 4], [4, 3], [4, 4]])
print(f"Dataset shape: {X.shape}")

# Fit GMM with 2 components
gmm = GaussianMixture(n_components=2)
gmm_labels = gmm.fit_predict(X)
print(f"Number of samples in each cluster: {np.bincount(gmm_labels)}")
visualize_gmm_clustering(X, gmm_labels, "GMM Clustering with Spherical Clusters")

# Limitation 2: GMMs are sensitive to initial placement
print("\nDemonstrating Limitation 2: Initialization Sensitivity")
print("Fitting GMM with random initialization...")
gmm_init_bad = GaussianMixture(n_components=2, init_params='random')
gmm_init_bad_labels = gmm_init_bad.fit_predict(X)
print(f"Number of samples in each cluster with random init: {np.bincount(gmm_init_bad_labels)}")
visualize_gmm_clustering(X, gmm_init_bad_labels, "GMM Clustering with Bad Initialization")

# Limitation 3: GMMs performance with high-dimensional data
print("\nDemonstrating Limitation 3: High-dimensional Data Challenge")
print("Generating high-dimensional random data...")
X_high_dim = np.random.rand(1000, 100)  # Generate high-dimensional data
print(f"High-dimensional dataset shape: {X_high_dim.shape}")

# Record time taken for high-dimensional clustering
import time
start_time = time.time()
gmm_high_dim = GaussianMixture(n_components=5)
gmm_high_dim_labels = gmm_high_dim.fit_predict(X_high_dim)
end_time = time.time()
print(f"Time taken for high-dimensional clustering: {end_time - start_time:.2f} seconds")
print(f"Number of samples in each cluster: {np.bincount(gmm_high_dim_labels)}")
visualize_gmm_clustering(X_high_dim[:, :2], gmm_high_dim_labels, "GMM Clustering with High-Dimensional Data (First 2 Dimensions)")
