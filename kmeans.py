import math
import random
import numpy as np
import copy

# IMPORTANT: DON'T CHANGE OR REMOVE THIS LINE
#            SO THAT YOUR RESULTS CAN BE VISUALLY SIMILAR
#            TO ONES GIVEN IN HOMEWORK FILES
random.seed(5710414)

class KMeans:
    def __init__(self, X, n_clusters, max_iterations=1000, epsilon=0.01, distance_metric="manhattan"):
        self.X = X
        self.n_clusters = n_clusters
        self.distance_metric = distance_metric
        self.clusters = []
        self.cluster_centers = []
        self.epsilon = epsilon
        self.max_iterations = max_iterations

    def fit(self):
        # TODO: Implement this function
        # Find clussters and cluster centers
        # Return nothing, but make sure that self.clusters and self.cluster_centers are filled
        if self.distance_metric == "manhattan":
            fit_helper(self, manhattan_dist)
        else:
            fit_helper(self, euclidean_dist)

    def predict(self, instance):
        # TODO: Implement this function
        # Return best cluster index for the given instance
        if self.distance_metric == "manhattan":
            return np.argmin(manhattan_dist(instance, self.cluster_centers))
        else:
            return np.argmin(euclidean_dist(instance, self.cluster_centers))


def fit_helper(self, dist_func):
    empty_clusters = []
    for cluster in range(self.n_clusters):
        self.cluster_centers.append(generate_random_color())
        empty_clusters.append([])
    for epoch in range(self.max_iterations):
        print("KMeans iteration: {}".format(epoch+1))
        halt = True
        self.clusters = copy.deepcopy(empty_clusters)
        for pixel in self.X:
            cluster_idx = np.argmin(dist_func(pixel, self.cluster_centers))
            self.clusters[cluster_idx].append(pixel)
        new_centers = []
        for cluster in self.clusters:
            if cluster:
                average = np.sum(cluster, 0)/len(cluster)
                new_centers.append((average[0], average[1], average[2]))
            else:
                new_centers.append((0, 0, 0))
        dist = dist_func(new_centers, self.cluster_centers)
        if (dist >= self.epsilon).any():
            halt = False
        self.cluster_centers = new_centers
        if halt:
            print("Epsilon boundary reached! Halting...")
            return
    print("Max iterations reached! Halting...")


def generate_random_color():
    return int(random.uniform(0, 256)), int(random.uniform(0, 256)), int(random.uniform(0, 256))


def manhattan_dist(x, y):
    return np.abs((x - np.array(y))[:, np.newaxis]).sum(axis=2)


def euclidean_dist(x, y):
    return np.sqrt(((x - np.array(y))[:, np.newaxis])**2).sum(axis=2)
