import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging

LOGGER = logging.getLogger(__name__)

class ClusteringWorkflow:
    def __init__(self, data, statistics_dataset, min_k=2, max_k=10):
        """
        Initialize the ClusteringWorkflow class.
        
        Parameters:
        - data (array-like): Input data for clustering.
        - statistics_dataset (xarray.Dataset): Dataset containing the statistics and coordinates.
        - min_k (int): Minimum number of clusters to consider.
        - max_k (int): Maximum number of clusters to consider.
        """
        self.data = data
        self.statistics_dataset = statistics_dataset
        self.min_k = min_k
        self.max_k = max_k

    def silhouette_analysis(self):
        silhouette_scores = []
        for k in range(self.min_k, self.max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(self.data)
            score = silhouette_score(self.data, cluster_labels)
            silhouette_scores.append(score)

        # Plot the Silhouette scores
        plt.plot(range(self.min_k, self.max_k + 1), silhouette_scores, 'bx-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis for Optimal k')
        plt.savefig("cluster.png")
        
        # Find the optimal k based on the highest Silhouette Score
        optimal_k_silhouette = silhouette_scores.index(max(silhouette_scores)) + self.min_k
        return silhouette_scores, optimal_k_silhouette

    def find_optimal_clusters(self):
        wcss = []
        for k in range(self.min_k, self.max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.data)
            wcss.append(kmeans.inertia_)

        # Plot the Elbow graph
        plt.plot(range(self.min_k, self.max_k + 1), wcss, 'bx-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('WCSS (Inertia)')
        plt.title('Elbow Method for Optimal k')
        plt.savefig("elbow.png")
        
        return wcss

    @staticmethod
    def select_optimal_k(silhouette_scores, wcss_scores, min_k):
        if len(silhouette_scores) != len(wcss_scores):
            raise ValueError("Silhouette scores and WCSS scores must have the same length")

        k_range = list(range(min_k, min_k + len(silhouette_scores)))

        # Normalize the Silhouette scores and invert WCSS scores
        norm_silhouette = (silhouette_scores - np.min(silhouette_scores)) / (np.max(silhouette_scores) - np.min(silhouette_scores))
        norm_wcss = (np.max(wcss_scores) - wcss_scores) / (np.max(wcss_scores) - np.min(wcss_scores))

        # Calculate variance and find optimal k
        variance = np.abs(norm_silhouette - norm_wcss)
        optimal_k_index = np.argmax(variance)
        optimal_k = k_range[optimal_k_index]
        
        return optimal_k

    def apply_kmeans(self, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.data)
        return cluster_labels

    def create_clustered_map(self, cluster_labels):
        clustered_map = cluster_labels.reshape(self.statistics_dataset.sizes['y'], self.statistics_dataset.sizes['x'])
        cluster_map = xr.DataArray(clustered_map, coords=[self.statistics_dataset['y'], self.statistics_dataset['x']], dims=['y', 'x'])
        cluster_map.plot()
        return cluster_map

    def plot_silhouette_vs_wcss(self):
        silhouette_scores, _ = self.silhouette_analysis()
        wcss = self.find_optimal_clusters()

        fig, ax1 = plt.subplots()

        # Plot Silhouette Scores on the left y-axis
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Silhouette Score', color='tab:blue')
        ax1.plot(range(self.min_k, self.max_k + 1), silhouette_scores, 'bx-', label='Silhouette Score', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Create a twin axis for the WCSS plot
        ax2 = ax1.twinx()
        ax2.set_ylabel('WCSS (Inertia)', color='tab:red')
        ax2.plot(range(self.min_k, self.max_k + 1), wcss, 'rx-', label='WCSS (Inertia)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        plt.title('Silhouette Score vs WCSS (Inertia) for Optimal k')
        fig.tight_layout()  
        plt.savefig("silhouette_vs_wcss.png")

    @staticmethod
    def plot_data_array(data_array, title='Data Array', cmap='viridis', vmin=None, vmax=None):
        plt.figure(figsize=(10, 6))
        data_array.plot(cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(title)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig("Clustered_Map.png")

    def full_clustering_workflow(self):
        silhouette_scores, _ = self.silhouette_analysis()
        wcss = self.find_optimal_clusters()

        optimal_k = self.select_optimal_k(silhouette_scores, wcss, self.min_k)
        LOGGER.info(f"Final Selected Optimal k: {optimal_k}")

        cluster_labels = self.apply_kmeans(n_clusters=optimal_k)
        clustered_map = self.create_clustered_map(cluster_labels)

        # Plot silhouette vs WCSS for visualization
        self.plot_silhouette_vs_wcss()
        
        # Plot final clustered map
        self.plot_data_array(clustered_map, title='Optimized Number of Clusters')

        return clustered_map
