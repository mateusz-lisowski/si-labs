import numpy as np


def calculate_distance(data_point, centroids):
    return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))


def initialize_centroids_forgy(data, k):
    return np.random.uniform(np.amin(data, axis=0), np.amax(data, axis=0), size=(k, data.shape[1]))


def initialize_centroids_kmeans_pp(data, k):
    centroids = [data[np.random.randint(data.shape[0])]]
    while len(centroids) < k:
        distances = np.array([min([np.linalg.norm(x - c) for c in centroids]) for x in data])
        probabilities = distances / distances.sum()
        cumulative_probabilities = probabilities.cumsum()
        r = np.random.rand()
        for j, p in enumerate(cumulative_probabilities):
            if r < p:
                i = j
                break
        centroids.append(data[i])
    return np.array(centroids)


def assign_to_cluster(data, centroid):
    result = []

    for data_point in data:
        distances = calculate_distance(data_point, centroid)
        cluster_num = np.argmin(distances)
        result.append(cluster_num)

    return np.array(result)


def update_centroids(data: np.ndarray, assignments: np.ndarray, centroids: np.ndarray):

    cluster_indices = []
    cluster_centers = []

    for i in range(len(centroids)):
        cluster_indices.append(np.argwhere(i == assignments))

    for i, indices in enumerate(cluster_indices):
        if len(indices) == 0:
            cluster_centers.append(centroids[i])
        else:
            cluster_centers.append(np.mean(data[indices], axis=0)[0])

    return np.array(cluster_centers)


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
        centroids = update_centroids(data, assignments, centroids)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments):  # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         

