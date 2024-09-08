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
