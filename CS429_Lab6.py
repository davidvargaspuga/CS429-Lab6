import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
style.use('ggplot')

# Works Cited:
# https://pythonprogramming.net/k-means-from-scratch-2-machine-learning-tutorial/?completed=/k-means-from-scratch-machine-learning-tutorial/

# Choose value for K
# Randomly select K featuresets to start as your centroids
# Calculate distance of all other featuresets to centroids
# Classify other featuresets as same as closest centroid
# Take mean of each class (mean of all featuresets by class), making that mean the new centroid
# Repeat steps 3-5 until optimized (centroids no longer moving)

class k_means:
    def __init__(self, k=8, tol = 0.00001, max_iter=1000) -> None:
        self.k = k
        self.tol = tol # tolerance value
        self.max_iter = max_iter
        self.main()



    def fit(self, data):
        # self.centroids = np.array([data[i] for i in range(self.k)])

        self.centroids = {}

        # random_sample_idxs = np.random.choice(data.shape, self.k, replace=False) 
        
        for i in range(self.k):
            self.centroids[i] = data[len(data) - 1 - i]

        for i in range(self.max_iter):
            
            print("iteration # ", i)
            self.classifications = {} # Inits dict for clusters

            # inits the keys of classifications dict as the centroids
            for i in range(self.k):
                self.classifications[i] = []

            # for values in data, find distances from centroids and then themin distance
            # add data point to the cluster it has min distance to, in classification dict
            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = (distances.index(min(distances)))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids) # saves centroids before changing them
            
            # for all clusters, finds average of data in cluster and makes that the new centroid
            for classification in self.classifications:
                self.centroids[classification]  = np.average(self.classifications[classification], axis=0)

            optimized = True 

            # stop if the new centroid makes a difference of 0.001 or less
            print("num of centroids", len(self.centroids))
            for c in self.centroids:
                print("c", c)
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if abs(np.sum( (current_centroid-original_centroid) / original_centroid*100.0 )) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break

            print("centroids", self.centroids)



    def main(self):
        print("in main")
        data = pd.read_csv('./synthetic/synthetic.txt', delimiter=' ', header=None, names=['X','Y'])
        data = data.to_numpy() # converts df to np arr
        print("data", type(data))
        # np.random.shuffle(data) 
        self.fit(data)
        # colors = ["g","r","c","b","m","y","burlywood", "chartreuse"]
        plt.title('Clustering')
        for centroid in self.centroids:
            print("{}, {}".format(self.centroids[centroid][0], self.centroids[centroid][1]))
            # plt.scatter(self.centroids[centroid][0], self.centroids[centroid][1], marker="o", color="k", s=150, linewidths=5)

        # for classification in self.classifications:
        #     color = colors[classification]
        #     for featureset in self.classifications[classification]:
        #         plt.scatter(featureset[0], featureset[1], color=color)

        # plt.savefig("file.png")
        # plt.show()
        



kmeans = k_means()

