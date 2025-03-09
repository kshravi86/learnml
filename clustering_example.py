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

# Create a side-by-side comparison of Hierarchical and K-Means clustering
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=hclust_labels, cmap='viridis')
plt.title("Hierarchical Clustering")  # Ward linkage method works well for spherical clusters

# K-Means often struggles with non-spherical clusters like this moon dataset
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
plt.title("K-Means Clustering")

plt.show()

# GMM is more flexible than K-Means as it can capture non-spherical clusters
# by modeling each cluster with its own covariance matrix
gmm = GaussianMixture(n_components=2, random_state=42)
gmm_labels = gmm.fit_predict(X)

# Plot GMM results - should handle moon shapes better than K-Means
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='viridis')
plt.title("Gaussian Mixture Model Clustering")
plt.show()

# Demonstrate how initialization affects GMM performance
# Using random initialization instead of k-means++ can lead to suboptimal results
gmm_init_bad = GaussianMixture(n_components=2, init_params='random', random_state=42)
gmm_init_bad_labels = gmm_init_bad.fit_predict(X)

# Plot GMM clustering results with bad initialization
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=gmm_init_bad_labels, cmap='viridis')
plt.title("Gaussian Mixture Model Clustering with Bad Initialization")
plt.show()

# This code is trying to demonstrate the strengths and weaknesses of different clustering algorithms (Hierarchical Clustering, K-Means, and Gaussian Mixture Model) on a non-linearly separable dataset. It shows how these algorithms perform in identifying the two moon-shaped clusters. Additionally, it illustrates the effect of initialization on the Gaussian Mixture Model's performance.

"""
Mathematical Explanations of the Clustering Algorithms:

1. Hierarchical Clustering (Ward's Method):
   - Minimizes the total within-cluster variance
   - Ward's criterion: minimize increase in sum of squares
   - Distance between clusters A and B:
     d(A,B) = √[(sum of squares of merged cluster) - (sum of squares of A) - (sum of squares of B)]
   
2. K-Means:
   - Objective: Minimize within-cluster sum of squares (WCSS)
   - Mathematical formulation:
     argmin_C Σ_{i=1}^k Σ_{x in c_i} ||x - μ_i||²
     where:
     - k is number of clusters
     - μ_i is the centroid of cluster c_i
     - ||x - μ_i|| is Euclidean distance

3. Gaussian Mixture Model (GMM):
   - Probability density function:
     p(x) = Σ_{k=1}^K π_k N(x|μ_k, Σ_k)
     where:
     - π_k are mixing coefficients (Σπ_k = 1)
     - N(x|μ_k, Σ_k) is Gaussian distribution with mean μ_k and covariance Σ_k
   - Uses Expectation-Maximization (EM) algorithm:
     E-step: Compute responsibilities
     γ(z_ik) = π_k N(x_i|μ_k, Σ_k) / Σ_j π_j N(x_i|μ_j, Σ_j)
     M-step: Update parameters
     μ_k = Σ_i γ(z_ik)x_i / Σ_i γ(z_ik)
     Σ_k = Σ_i γ(z_ik)(x_i - μ_k)(x_i - μ_k)ᵀ / Σ_i γ(z_ik)
     π_k = Σ_i γ(z_ik) / N

Key Differences:
- Hierarchical: No assumptions about cluster shape, builds hierarchy
- K-Means: Assumes spherical clusters, sensitive to initialization
- GMM: Flexible cluster shapes, probabilistic membership, but more complex
"""
