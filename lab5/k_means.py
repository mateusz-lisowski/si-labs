import numpy as np


def initialize_centroids_forgy(data, k):
    return np.random.uniform(np.amin(data, axis=0), np.amax(data, axis=0), size=(k, data.shape[1]))


def initialize_centroids_kmeans_pp(data, k):
    # TODO implement kmeans++ initialization
    return None


def assign_to_cluster(data, centroid):
    # TODO find the closest cluster for each data point
    return None


def update_centroids(data, assignments):
    # TODO find new centroids based on the assignments
    return None


def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))


def k_means(data, num_centroids, kmeans_plus_plus=False):
    # centroids initialization
    if kmeans_plus_plus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else: 
        centroids = initialize_centroids_forgy(data, num_centroids)
    
    assignments = assign_to_cluster(data, centroids)
    for i in range(100):    # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments):  # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         

