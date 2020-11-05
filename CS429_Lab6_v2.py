import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
style.use('ggplot')

# Choose value for K
# Randomly select K featuresets to start as your centroids
# Calculate distance of all other featuresets to centroids
# Classify other featuresets as same as closest centroid
# Take mean of each class (mean of all featuresets by class), making that mean the new centroid
# Repeat steps 3-5 until optimized (centroids no longer moving)
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans:
    def __init__(self, k=8, max_iters=1000000, plot_steps=True) -> None:
        self.k = k
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        self.clusters = [[] for _ in range(self.k)]
        self.centroids = []
        
        self.true_centroids = pd.read_csv("synthetic/synthetic-gt.txt", header=None)
        self.true_centroids = self.true_centroids.values

    def fit(self, data):
        self.data = data
        self.n_samples, self.n_features = data.shape

        random_sample_indices = np.random.choice(self.n_samples, self.k, replace=False)
        print("RSIs", random_sample_indices)

        self.centroids = [self.data[idx] for idx in random_sample_indices]

        for item in self.centroids:
            print("PRE-LEARNING", item)
        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(self.centroids)

            # if self.plot_steps:
            #     self.plot()

            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            # if self.plot_steps:
            #     self.plot()
                        
            if self._is_converged(centroids_old, self.centroids):
                for item in self.centroids:
                    print("GUESSED ", item)

                for item in self.true_centroids:
                    print("ACTUAL ", item)
                break

        if self.plot_steps:
            self.plot()
        return self._get_cluster_lables(self.clusters)

    def _get_cluster_lables(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.data):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.k, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.data[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids
    
    def _is_converged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.k)]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, index in enumerate(self.clusters):
            point = self.data[index].T
            ax.scatter(*point)
        for point in self.centroids:
            ax.scatter(*point, marker="x", color='black', linewidth=2)

        plt.show()

k = KMeans()
data = pd.read_csv('./synthetic/synthetic.txt', delimiter=' ', header=None, names=['X','Y'])
data = data.to_numpy() # converts df to np arr
y_pred = k.fit(data)

print(y_pred)