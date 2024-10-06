import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture

# Generate sample data using make_moons function
X, _ = make_moons(n_samples=200, noise=0.05)

# Perform Hierarchical Clustering using AgglomerativeClustering
hclust = AgglomerativeClustering(n_clusters=2, linkage='ward')
hclust_labels = hclust.fit_predict(X)

# Perform K-Means Clustering using KMeans
kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Plot the clustering results
plt.figure(figsize=(10, 5))

# Plot Hierarchical Clustering results
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=hclust_labels, cmap='viridis')
plt.title("Hierarchical Clustering")

# Plot K-Means Clustering results
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
plt.title("K-Means Clustering")

plt.show()

# Perform Gaussian Mixture Model Clustering
gmm = GaussianMixture(n_components=2, random_state=42)
gmm_labels = gmm.fit_predict(X)

# Plot GMM clustering results
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='viridis')
plt.title("Gaussian Mixture Model Clustering")
plt.show()

# Perform Gaussian Mixture Model Clustering with random initialization
gmm_init_bad = GaussianMixture(n_components=2, init_params='random', random_state=42)
gmm_init_bad_labels = gmm_init_bad.fit_predict(X)

# Plot GMM clustering results with bad initialization
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=gmm_init_bad_labels, cmap='viridis')
plt.title("Gaussian Mixture Model Clustering with Bad Initialization")
plt.show()

# This code is trying to demonstrate the strengths and weaknesses of different clustering algorithms (Hierarchical Clustering, K-Means, and Gaussian Mixture Model) on a non-linearly separable dataset. It shows how these algorithms perform in identifying the two moon-shaped clusters. Additionally, it illustrates the effect of initialization on the Gaussian Mixture Model's performance.
