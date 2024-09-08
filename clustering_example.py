import numpy as np  # Import numpy library
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from sklearn.cluster import AgglomerativeClustering, KMeans  # Import clustering algorithms
from sklearn.datasets import make_moons  # Import dataset generator

# Generate sample data using make_moons function
# This will create a moon-shaped dataset with 200 samples and 0.05 noise
X, _ = make_moons(n_samples=200, noise=0.05)

# Perform Hierarchical Clustering using AgglomerativeClustering
# with 2 clusters and ward linkage
hclust = AgglomerativeClustering(n_clusters=2, linkage='ward')
hclust_labels = hclust.fit_predict(X)  # Fit the model and predict cluster labels

# Perform K-Means Clustering using KMeans
# with 2 clusters
kmeans = KMeans(n_clusters=2)
kmeans_labels = kmeans.fit_predict(X)  # Fit the model and predict cluster labels

# This code demonstrates the difference between Hierarchical Clustering and K-Means Clustering
# on a moon-shaped dataset, showcasing their strengths and weaknesses in clustering
# Plot the clustering results
plt.figure(figsize=(10, 5))  # Create a figure with size 10x5

# Plot Hierarchical Clustering results
plt.subplot(1, 2, 1)  # Create a subplot for Hierarchical Clustering
plt.scatter(X[:, 0], X[:, 1], c=hclust_labels)  # Scatter plot with cluster labels
plt.title("Hierarchical Clustering")  # Set title for the subplot

# Plot K-Means Clustering results
plt.subplot(1, 2, 2)  # Create a subplot for K-Means Clustering
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels)  # Scatter plot with cluster labels
plt.title("K-Means Clustering")  # Set title for the subplot

plt.show()

# Limitations of Gaussian Mixture Models (GMMs) for clustering
# ======================================================

# 1. GMMs assume spherical clusters
# GMMs are sensitive to the scale of the features, which can lead to poor clustering results
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=2)
gmm_labels = gmm.fit_predict(X)

# Plot GMM clustering results
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=gmm_labels)
plt.title("Gaussian Mixture Model Clustering")
plt.show()

# 2. GMMs are sensitive to the initial placement of centroids
# GMMs can get stuck in local optima, leading to poor clustering results
gmm_init_bad = GaussianMixture(n_components=2, init_params='random')
gmm_init_bad_labels = gmm_init_bad.fit_predict(X)

# Plot GMM clustering results with bad initialization
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=gmm_init_bad_labels)
plt.title("Gaussian Mixture Model Clustering with Bad Initialization")
plt.show()

# 3. GMMs can be computationally expensive for high-dimensional data
# GMMs have a time complexity of O(n*d*k), where n is the number of samples, d is the number of features, and k is the number of components
# This can make them impractical for large datasets
