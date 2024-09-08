import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Limitation 1: GMMs assume spherical clusters
# GMMs are sensitive to the scale of the features, which can lead to poor clustering results
X = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [3, 3], [3, 4], [4, 3], [4, 4]])
gmm = GaussianMixture(n_components=2)
gmm_labels = gmm.fit_predict(X)

plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=gmm_labels)
plt.title("GMM Clustering with Spherical Clusters")
plt.show()

# Limitation 2: GMMs are sensitive to the initial placement of centroids
# GMMs can get stuck in local optima, leading to poor clustering results
gmm_init_bad = GaussianMixture(n_components=2, init_params='random')
gmm_init_bad_labels = gmm_init_bad.fit_predict(X)

plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=gmm_init_bad_labels)
plt.title("GMM Clustering with Bad Initialization")
plt.show()

# Limitation 3: GMMs can be computationally expensive for high-dimensional data
# GMMs have a time complexity of O(n*d*k), where n is the number of samples, d is the number of features, and k is the number of components
# This can make them impractical for large datasets
X_high_dim = np.random.rand(1000, 100)  # Generate high-dimensional data
gmm_high_dim = GaussianMixture(n_components=5)
gmm_high_dim_labels = gmm_high_dim.fit_predict(X_high_dim)

plt.figure(figsize=(5, 5))
plt.scatter(X_high_dim[:, 0], X_high_dim[:, 1], c=gmm_high_dim_labels)
plt.title("GMM Clustering with High-Dimensional Data")
plt.show()
