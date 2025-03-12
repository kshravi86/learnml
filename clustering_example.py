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

"""
Detailed Mathematical Framework Behind Each Clustering Algorithm:

1. Hierarchical Clustering (Ward's Method):
   • Core Concept: Minimizes within-cluster variance V_w
   • Mathematical Definition:
     V_w = Σ_{i=1}^k Σ_{x∈C_i} (x - μ_i)ᵀ(x - μ_i)
     where C_i is cluster i and μ_i is its centroid

2. K-Means Algorithm:
   • Objective Function: J = Σ_{i=1}^k Σ_{x∈C_i} ||x - μ_i||²
   • Update Rules:
     - Cluster Assignment: C_i = {x : ||x - μ_i|| ≤ ||x - μ_j|| ∀j}
     - Centroid Update: μ_i = (1/|C_i|) Σ_{x∈C_i} x

3. Gaussian Mixture Model:
   • Full Probability Model:
     p(x) = Σ_{k=1}^K π_k N(x|μ_k, Σ_k)
   • Log-likelihood:
     L = Σ_{i=1}^N log(Σ_{k=1}^K π_k N(x_i|μ_k, Σ_k))
   • Complete EM Update Equations:
     - E-step: γ(z_ik) = π_k N(x_i|μ_k, Σ_k) / Σ_j π_j N(x_i|μ_j, Σ_j)
     - M-step:
       * N_k = Σ_i γ(z_ik)
       * π_k_new = N_k / N
       * μ_k_new = (1/N_k) Σ_i γ(z_ik)x_i
       * Σ_k_new = (1/N_k) Σ_i γ(z_ik)(x_i - μ_k_new)(x_i - μ_k_new)ᵀ
"""

"""
Comprehensive Mathematical Theory Behind Clustering Algorithms:

1. HIERARCHICAL CLUSTERING (Ward's Method)
====================================
• Linkage Function (Ward's):
  L(C_i, C_j) = √[(|C_i|·|C_j|)/(|C_i|+|C_j|)] ||centroid(C_i) - centroid(C_j)||₂
  
• Dissimilarity Update Formula:
  d(i∪j,k) = [(n_i+n_k)d(i,k) + (n_j+n_k)d(j,k) - n_k·d(i,j)] / (n_i+n_j+n_k)
  where n_x is size of cluster x

2. K-MEANS ALGORITHM
================
• Mathematical Optimization Problem:
  minimize_{C} Σ_{k=1}^K Σ_{x∈C_k} ||x - μ_k||²
  
• Lloyd's Algorithm Steps:
  1) Assignment: z_i = argmin_k ||x_i - μ_k||²
  2) Update: μ_k = (1/|C_k|) Σ_{x∈C_k} x
  
• Convergence Properties:
  - Monotonically decreasing objective
  - Finite number of partitions ensures termination

3. GAUSSIAN MIXTURE MODEL
=====================
• Full Probabilistic Model:
  p(x) = Σ_{k=1}^K π_k N(x|μ_k, Σ_k)
  where N(x|μ,Σ) = (2π)^(-d/2)|Σ|^(-1/2)exp(-1/2(x-μ)ᵀΣ⁻¹(x-μ))

• EM Algorithm Complete Derivation:
  E-step (Posterior Probability):
    γ(z_ik) = p(z_k=1|x_i) = π_k N(x_i|μ_k,Σ_k) / Σ_j π_j N(x_i|μ_j,Σ_j)
  
  M-step (Parameter Updates):
    N_k = Σ_i γ(z_ik)
    π_k_new = N_k/N
    μ_k_new = (1/N_k) Σ_i γ(z_ik)x_i
    Σ_k_new = (1/N_k) Σ_i γ(z_ik)(x_i-μ_k_new)(x_i-μ_k_new)ᵀ

• Log-Likelihood:
  L = Σ_i log(Σ_k π_k N(x_i|μ_k,Σ_k))
"""

"""
COMPLETE MATHEMATICAL FOUNDATIONS OF CLUSTERING ALGORITHMS

A. THEORETICAL FOUNDATIONS
=========================
1. Distance Metrics
   • Euclidean: d(x,y) = √(Σ(x_i - y_i)²)
   • Mahalanobis: d(x,y) = √((x-y)ᵀS⁻¹(x-y))
   • Information Theoretic: KL(p||q) = Σp(x)log(p(x)/q(x))

2. Optimization Theory
   • Convex Optimization: min f(x) subject to x ∈ C
   • Jensen's Inequality: E[f(X)] ≥ f(E[X]) for convex f
   • Lagrange Multipliers: L(x,λ) = f(x) + λg(x)

B. ALGORITHM-SPECIFIC MATHEMATICS
===============================
1. Hierarchical Clustering
   • Lance-Williams Formula:
     d(i∪j,k) = α_i·d(i,k) + α_j·d(j,k) + β·d(i,j) + γ|d(i,k) - d(j,k)|
   • Ward's Criterion:
     α_i = (n_i + n_k)/(n_i + n_j + n_k)
     β = -n_k/(n_i + n_j + n_k)
     γ = 0

2. K-Means
   • Expectation Step:
     z_ik = 1 if k = argmin_j ||x_i - μ_j||²
     z_ik = 0 otherwise
   • Maximization Step:
     μ_k = (Σ_i z_ik x_i)/(Σ_i z_ik)
   • Convergence Rate:
     O(nkdi) where n=samples, k=clusters, d=dimensions, i=iterations

3. Gaussian Mixture Models
   • Multivariate Gaussian:
     N(x|μ,Σ) = (2π)^(-d/2)|Σ|^(-1/2)exp(-1/2(x-μ)ᵀΣ⁻¹(x-μ))
   • EM Algorithm:
     Q(θ|θᵗ) = E_Z[log p(X,Z|θ)|X,θᵗ]
   • ELBO (Evidence Lower BOund):
     L(q,θ) = E_q[log p(X,Z|θ)] - E_q[log q(Z)]

C. CONVERGENCE ANALYSIS
======================
1. K-Means:
   • Guaranteed local convergence
   • NP-hard globally
   • Bounded iterations: O(n^(kd))

2. EM Algorithm:
   • Monotonic convergence: L(θ^(t+1)) ≥ L(θ^(t))
   • Linear convergence rate near optima
   • Sensitive to initialization
"""

# Print detailed mathematical explanations
def print_mathematical_details():
    print("\nADVANCED MATHEMATICAL ANALYSIS OF CLUSTERING ALGORITHMS")
    print("="*60)
    
    print("\n1. HIERARCHICAL CLUSTERING - WARD'S METHOD")
    print("-"*45)
    print("• Energy Function:")
    print("  E = Σ_{C∈clusters} Σ_{x∈C} ||x - μ_C||²")
    print("• Merge Cost:")
    print("  ΔE(C_i,C_j) = (n_i·n_j)/(n_i+n_j) ||μ_i - μ_j||²")
    print("• Information Theory Perspective:")
    print("  - Minimizes information loss during merging")
    print("  - Related to decrease in Shannon entropy")
    
    print("\n2. K-MEANS - OPTIMIZATION THEORY")
    print("-"*45)
    print("• Objective Function (Vector Quantization):")
    print("  J(C,μ) = Σ_{k=1}^K Σ_{x∈C_k} ||x - μ_k||²")
    print("• Convergence Properties:")
    print("  - NP-hard problem in general")
    print("  - Local convergence in O(nKdi) iterations")
    print("    where: n=samples, K=clusters, d=dimensions, i=iterations")
    
    print("\n3. GAUSSIAN MIXTURE MODEL - PROBABILISTIC FRAMEWORK")
    print("-"*45)
    print("• Complete Data Log-Likelihood:")
    print("  l_c(θ) = Σ_i Σ_k z_ik[log π_k + log N(x_i|μ_k,Σ_k)]")
    print("• EM Convergence Properties:")
    print("  - Monotonic increase: L(θ^(t+1)) ≥ L(θ^(t))")
    print("  - Q-function: Q(θ|θ^(t)) = E_Z[log p(X,Z|θ)|X,θ^(t)]")
    print("• Relationship to K-means:")
    print("  - K-means is limiting case of GMM as covariances → 0")
    print("  - GMM provides soft assignments via responsibilities")

def print_detailed_math_analysis():
    print("\nCOMPREHENSIVE MATHEMATICAL ANALYSIS OF CLUSTERING")
    print("="*60)
    
    print("\n1. OPTIMIZATION FUNDAMENTALS")
    print("-"*45)
    print("• Gradient Descent Update:")
    print("  θ_t+1 = θ_t - η∇L(θ_t)")
    print("• Expectation-Maximization:")
    print("  Q(θ|θᵗ) = E_Z[log p(X,Z|θ)|X,θᵗ]")
    
    print("\n2. CLUSTERING METRICS")
    print("-"*45)
    print("• Silhouette Score:")
    print("  s(i) = (b(i) - a(i))/max(a(i),b(i))")
    print("• Calinski-Harabasz Index:")
    print("  CH = [tr(B_k)/(k-1)]/[tr(W_k)/(n-k)]")
    
    print("\n3. INFORMATION THEORY")
    print("-"*45)
    print("• Mutual Information:")
    print("  I(X;Y) = Σ p(x,y)log(p(x,y)/p(x)p(y))")
    print("• Entropy Reduction:")
    print("  ΔH = H(X) - Σ_k (n_k/n)H(X_k)")
    
    print("\n4. PROBABILISTIC BOUNDS")
    print("-"*45)
    print("• VC-Dimension Bound:")
    print("  R(h) ≤ R_emp(h) + √(VC·log(2n/VC)/n)")
    print("• PAC Learning Bound:")
    print("  P(|R(h)-R_emp(h)| ≤ ε) ≥ 1-δ")

if __name__ == "__main__":
    print_mathematical_details()
    print_detailed_math_analysis()

"""
USE CASES FOR EACH CLUSTERING ALGORITHM

1. Hierarchical Clustering Use Cases:
   - Taxonomic classification in biology (organizing species into hierarchies)
   - Document organization and topic hierarchy creation
   - Customer segmentation with nested groups
   - Social network analysis and community detection
   - Gene expression clustering in bioinformatics

2. K-Means Clustering Use Cases:
   - Market segmentation with well-defined, spherical clusters
   - Image compression (color quantization)
   - Anomaly detection in spherical data distributions
   - Customer segmentation with equal-sized groups
   - Data preprocessing for dimensionality reduction

3. Gaussian Mixture Model (GMM) Use Cases:
   - Speech recognition and speaker identification
   - Financial market regime detection
   - Complex pattern recognition in astronomy
   - Behavior modeling in robotics
   - Image segmentation with overlapping regions

Selection Guidelines:
- Use Hierarchical when: You need a hierarchy, don't know cluster count, or have small datasets
- Use K-Means when: You have spherical clusters, need speed, or have large datasets
- Use GMM when: You have overlapping clusters, need probability scores, or have complex shapes
"""
