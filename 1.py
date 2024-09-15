import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def compute_objective(X, cluster_labels, cluster_centers):

    #Calculate the k-means objective function, sum of squared distances of points to their cluster center.
    total_distance = 0

    for i, center in enumerate(cluster_centers):
        points = X[cluster_labels == i]
        total_distance += np.sum((points - center) ** 2)
        
    return total_distance


def Strategy_1(X, num_clusters):
    #Select 'num_clusters' random samples from X to serve as initial cluster centers.

    random_indices = np.random.choice(X.shape[0], num_clusters, replace=False)

    return X[random_indices]


def Strategy_2(X, num_clusters):
    #Select initial centers with the first being random and subsequent centers chosen to maximize distance to all previous centers.
    centers = []  

    first_index = np.random.choice(X.shape[0])  
    centers.append(X[first_index])

    for _ in range(1, num_clusters):

        distances = np.array([np.min([np.linalg.norm(x - center) for center in centers]) for x in X])
        new_center_index = np.argmax(distances)
        centers.append(X[new_center_index])
      
    return np.array(centers)

def assign_clusters(X, centers):
    #Assign each point in X to the nearest center, returning the cluster index for each point.

    distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)

    return np.argmin(distances, axis=1)

def k_means_clustering(X, initial_centers, max_iterations=100, tolerance=1e-4):
    #Perform k-means clustering, adjusting centers until convergence or reaching the maximum number of iterations.
    centers = initial_centers

    for _ in range(max_iterations):
        clusters = assign_clusters(X, centers)
        new_centers = np.array([X[clusters == i].mean(axis=0) for i in range(len(centers))])
        if np.linalg.norm(new_centers - centers) < tolerance:
            break
        centers = new_centers

    return centers

def load_data(filepath):
    #Load .mat file and return the 'AllSamples' data matrix.

    data = sio.loadmat(filepath)

    return data['AllSamples']

# Load the data
X = load_data("AllSamples.mat")

# Execute clustering using each initialization strategy
strategies = [Strategy_1, Strategy_2]
for strategy in strategies:
    for iteration in range(2):  # Perform two iterations for each strategy
        objective_values = []
        for k in range(2, 11):  # Test cluster sizes from 2 to 10
            initial_centers = strategy(X, k)
            final_centers = k_means_clustering(X, initial_centers)
            cluster_labels = assign_clusters(X, final_centers)
            objective = compute_objective(X, cluster_labels, final_centers)
            objective_values.append(objective)

        plt.plot(range(2, 11), objective_values, marker='o')
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Objective Function Value")
        plt.title(f"{iteration + 1} Iteration with {strategy.__name__}")
        plt.savefig(f"{strategy.__name__}_{iteration + 1}.png")
        plt.show()
