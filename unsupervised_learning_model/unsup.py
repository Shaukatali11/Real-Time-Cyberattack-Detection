import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler




# Define unsupervised learning algorithms
algorithms = {
    "K-Means": KMeans(n_clusters=3, random_state=42),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    "Agglomerative Clustering": AgglomerativeClustering(n_clusters=3),
    "Gaussian Mixture": GaussianMixture(n_components=3, random_state=42),
    "PCA": PCA(n_components=2),
    "ICA": FastICA(n_components=2, random_state=42),
    "t-SNE": TSNE(n_components=2, random_state=42)
}



# Standardize the dataset (important for clustering and dimensionality reduction)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)




from sklearn.metrics import adjusted_rand_score, silhouette_score

# Assuming 'y_test' and 'X_scaled' are now aligned
for algo_name, algo in algorithms.items():
    start_time = time.time()
    
    # Fit the algorithm to the scaled data
    if algo_name in ["PCA", "ICA"]:
        transformed_data = algo.fit_transform(X_scaled)
        # For PCA, calculate explained variance
        if algo_name == "PCA":
            explained_variance = np.sum(algo.explained_variance_ratio_)
            accuracy = explained_variance  # Use explained variance as a proxy for accuracy
        else:
            accuracy = None  # ICA doesn't have a clear accuracy metric
    else:
        # Fit clustering algorithms
        algo.fit(X_scaled)
        if hasattr(algo, 'labels_'):
            labels = algo.labels_  # For K-Means, DBSCAN, and Agglomerative Clustering
        else:
            labels = algo.predict(X_scaled)  # For Gaussian Mixture Model
        
        # If we have true labels, calculate ARI or Silhouette Score
        if 'y_test' in locals():  # Assuming y_test is available
            # Compute Adjusted Rand Index (ARI) for clustering accuracy
            accuracy = adjusted_rand_score(y_test, labels)  # You can replace y_test with the actual true labels
        else:
            # If no true labels are available, use Silhouette Score
            accuracy = silhouette_score(X_scaled, labels)

    end_time = time.time()
    detection_time = end_time - start_time

    # Store results
    results.append({
        "Algorithm": algo_name,
        "Accuracy": round(accuracy, 4) if accuracy is not None else "N/A",  # Display "N/A" for dimensionality reduction
        "Detection Time (seconds)": round(detection_time, 4)
    })

# Create a DataFrame for results
results_df = pd.DataFrame(results)
print(results_df)

