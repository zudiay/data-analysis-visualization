import random
import sys
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons


# generates initial random centroid locations within the data range, returns the list of centroid locations
def generate_random_centroid_locations(k: int, nodes: List[List[float]]) -> List[List[float]]:
    min_x = np.min(nodes, axis=0)[0]
    min_y = np.min(nodes, axis=0)[1]
    max_x = np.max(nodes, axis=0)[0]
    max_y = np.max(nodes, axis=0)[1]

    centroids = []
    for i in range(0, k):
        # for each cluster, generate a random x and random y
        centroid_x = random.uniform(min_x, max_x)
        centroid_y = random.uniform(min_y, max_y)
        centroids.append([centroid_x, centroid_y])
    return centroids


# calculates and returns euclidean distance between two nodes
def calculate_euclidean(node: List[float], centroid: List[float]) -> float:
    dist = (node[0] - centroid[0]) ** 2 + (node[1] - centroid[1]) ** 2
    return dist


# finds the nearest centroid to a node, given the list of centroids, returns the index of it
def find_nearest_centroid(node: List[float], centroids: List[List[float]]) -> int:
    nearest_centroid = 0
    nearest_distance = calculate_euclidean(node, centroids[0])
    for i in range(0, len(centroids)):
        dist = calculate_euclidean(node, centroids[i])
        if dist < nearest_distance:
            nearest_distance = dist
            nearest_centroid = i
    return nearest_centroid


# assigns nodes to nearest centroids to them, returns the list of nearest centroid indexes corresponding to each node
def assign_nodes_to_nearest_centroids(nodes: List[List[float]], centroids: List[List[float]]) -> List[int]:
    assigned_centroids = []
    for node in nodes:
        nearest_centroid = find_nearest_centroid(node, centroids)
        assigned_centroids.append(nearest_centroid)
    return assigned_centroids


# calculates and returns the within cluster variation, given a list of nodes belonging to a cluster and the centroid
def calculate_within_cluster_variation(cluster_nodes: List[List[float]], cluster_centroid: List[float]) -> float:
    objective = 0
    for i in range(0, len(cluster_nodes)):
        objective += calculate_euclidean(cluster_nodes[i], cluster_centroid)
    return objective


# calculates and returns the locations of centroids, given the list of nodes and their corresponding clusters
# also returns the objective function value with the given node-cluster assignments and calculated centroid locations
def calculate_centroid_locations(k: int, nodes: List[List[float]], assigned_centroids: List[int]) -> Tuple[
    Union[List[List[float]], bool], float]:
    centroid_locations = []
    objective = 0
    for i in range(0, k):
        centroid_cluster = []
        # traverse all nodes, if a node is assigned to centroid i, add it to the list of nodes for that centroid
        for j in range(0, len(nodes)):
            if assigned_centroids[j] == i:
                centroid_cluster.append(nodes[j])
        # after finding all nodes belonging to cluster i, calculate the center point of the nodes
        # if the cluster is empty, stop the algotihm
        if len(centroid_cluster) == 0:
            return False, 0
        centroid_x = np.average(centroid_cluster, axis=0)[0]
        centroid_y = np.average(centroid_cluster, axis=0)[1]
        # after determining the new centroid for cluster i, calculate the objective function value
        centroid_locations.append([centroid_x, centroid_y])
        objective += calculate_within_cluster_variation(centroid_cluster, [centroid_x, centroid_y])

    return centroid_locations, objective


# given a list of nodes and the assigned centroids, plots the clusters in different colors
def plot_clusters(k: int, X_np: np.array, assignment_np: np.array):
    colors = ['lightblue', 'yellowgreen', 'pink', 'coral', 'mediumpurple',
              'gold', 'whitesmoke', 'wheat', 'tomato', 'springgreen',
              'sienna', 'sandybrown', 'rosybrown', 'plum', 'peru',
              'orange', 'orchid', 'olive', 'midnightblue', 'magenta']
    for i in range(0, k):
        plt.scatter(
            X_np[assignment_np == i, 0], X_np[assignment_np == i, 1],
            c=colors[i], edgecolor='black',
            label=f'cluster {i}'
        )


# plots the centroids
def plot_centroids(centroids: np.array):
    plt.scatter(
        centroids[:, 0], centroids[:, 1],
        c='black', edgecolor='red',
        label='centroids'
    )


# plots the change of objective function value against time
def plot_objective(objective_function_values: np.array, is_elbow: bool = False, k: int = 0):
    plt.plot(range(1, len(objective_function_values) + 1), objective_function_values, marker='o')
    if is_elbow:
        plt.xlabel(f'k value')
        plt.title(f'Elbow Function')
    else:
        plt.xlabel('Iteration')
        plt.title(f'Objective Function Values for K={k}')
    plt.ylabel(f'Objective Function Value')
    plt.show()


# given the iteration count, reproduces the k-means algorithm, plots the first 3 and last iterations
def plot_k_means(K, X, centroid_locations, iterations):
    iteration = 0
    while 1:
        centroid_assignments = assign_nodes_to_nearest_centroids(nodes=X, centroids=centroid_locations)

        # plot the first 3 iterations
        if iteration <= 3:
            plot_clusters(K, np.array(X), np.array(centroid_assignments))
            plot_centroids(np.array(centroid_locations))
            if iteration == 0:
                plt.title(f'Initial Cluster Centers for K={K}')
            else:
                plt.title(f'Clusters and Centroids for K={K} Iteration {iteration}')
            plt.legend()
            plt.show()

        iteration += 1
        centroid_locations, objective = calculate_centroid_locations(k=K, nodes=X,
                                                                     assigned_centroids=centroid_assignments)

        # stop the algorithm if the objective function stops improving, plot the last iteration
        if iteration == iterations:
            plot_clusters(K, np.array(X), np.array(centroid_assignments))
            plot_centroids(np.array(centroid_locations))
            plt.title(f'Clusters and Centroids for K={K} for the Last Iteration ({iteration})')
            plt.legend()
            plt.show()
            break


# plots the skicit-learn library's output for a given dataset and k value
def plot_skicit(X, K):
    X = np.array(X)
    k_means = KMeans(n_clusters=K, init='random')
    y_km = k_means.fit_predict(X)
    plot_clusters(K, X, y_km)
    plot_centroids(k_means.cluster_centers_)
    plt.title(f'Clusters and Centroids by scikit-learn')
    plt.legend()
    plt.show()


def k_means(K, X, centroid_locations) -> Tuple[bool, List[float], int]:
    objective_function_values = []
    objective = 0
    iteration = 0
    while 1:
        iteration += 1

        # assign given nodes to nearest centroids, then, recalculate the locations of centroid for each cluster
        centroid_assignments = assign_nodes_to_nearest_centroids(nodes=X, centroids=centroid_locations)
        prev_objective = objective
        centroid_locations, objective = calculate_centroid_locations(k=K, nodes=X,
                                                                     assigned_centroids=centroid_assignments)

        # if a cluster is empty, the initial cluster centers should be changed, restart the algorithm
        if not centroid_locations:
            return False, [], 0

        objective_function_values.append(objective)

        # stop the algorithm if the objective function stops improving, return
        if iteration > 3 and prev_objective - objective < 0.0001:
            return True, objective_function_values, iteration
        # return the objective function values for each iteration and total iteration count


def main(convex):
    # create dataset and random initial centroid locations
    if convex:
        X, y = make_blobs(n_samples=500, centers=3, cluster_std=1.7, random_state=17)
    else:
        X, y = make_moons(n_samples=500, noise=0.1, random_state=17)
    plt.scatter(np.array(X)[:, 0], np.array(X)[:, 1], color='gray', edgecolor='black')
    plt.title(f'Dataset')
    plt.show()

    final_objective_function_values = []  # keep objective function values for each k to be able to detect the elbow
    initial_random_centroid_locations_for_k = {}  # keep the initial random centroid locations for each k to be able to reproduce
    iterations_for_k = {}  # keep the iteration count for each k
    objective_function_values_for_k = {}  # keep the objective function change for each k to be able to plot

    for K in range(1, 15):
        while 1:
            # start the k means algorithm with random initial centroid locations
            centroid_locations = generate_random_centroid_locations(K, X.tolist())
            initial_random_centroid_locations_for_k[K] = centroid_locations
            valid, objective_array_for_k, iteration_for_k = k_means(K, X, centroid_locations)
            if valid:
                break
        # if k_means has terminated without problem, save the results
        iterations_for_k[K] = iteration_for_k
        objective_function_values_for_k[K] = objective_array_for_k
        final_objective_function_values.append(objective_array_for_k[-1])

    # after calculating termination objective function values for each k, detect the best k using elbow method
    # the elbow occurs where the most crucial change in the objective function value appears
    # this point can be thought as the point that has the biggest difference ratio with the previous point
    plot_objective(final_objective_function_values, is_elbow=True)
    elbow = 1
    max_ratio = 0
    for i, val in enumerate(final_objective_function_values):
        if i == 0:
            continue
        ratio = (final_objective_function_values[i - 1] - final_objective_function_values[i]) / \
                final_objective_function_values[i]
        if ratio > max_ratio:
            elbow = i
            max_ratio = ratio

    # after deciding the best k, plot iterations of k means, the objective function change and skicit's output
    best_k = elbow + 1
    plot_k_means(best_k, X, initial_random_centroid_locations_for_k[best_k], iterations_for_k[best_k])
    plot_objective(objective_function_values_for_k[best_k], k=best_k)
    plot_skicit(X, best_k)


if __name__ == "__main__":
    convex = True if len(sys.argv) == 1 else (False if sys.argv[1] == 'False' else True)
    main(convex=convex)  # use True for convex dataset
