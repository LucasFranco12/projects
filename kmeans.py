import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import numpy as np

def kmeans(data, k):
    flip = False
    labels = np.zeros((30, 1), dtype='int')
    # actual kmeans algorithm
    n, d = data.shape
    centroids = np.random.rand(k, d)
    dist = np.zeros((n, k), dtype='float')

    while not flip:
        
        for c in range(0, k):
            for a in range(0, n):
                dist[a, c] = np.linalg.norm(centroids[c, :] - data[a, :])
        # assign the closest centroid to each sample
        for a in range(0, n):
            mindist = dist[a, 0]
            min_c = 0
            for c in range(0, k):
                if mindist > dist[a, c]:
                    mindist = dist[a, c]
                    min_c = c
            labels[a] = min_c
        # for each cluster
        new_centroids = np.zeros((k, d), dtype='float')
        for c in range(0, k):
            count = 0
            for a in range(0, n):
                if labels[a] == c:
                    new_centroids[c, :] += data[a, :]
                    count += 1
            if count > 0:
                new_centroids[c, :] /= count
            print(new_centroids)
        
        if np.allclose(centroids, new_centroids, atol=1e-4):
            flip = True

        centroids = new_centroids
    return labels, centroids, new_centroids

def main():
    # Set a random seed for reproducibility
    #np.random.seed(42)
    
    data = np.random.rand(30, 2)
    labels, centroids, new_centroids = kmeans(data, 3)
    
    # Create a figure and customize it
    plt.figure(figsize=(10, 6))
    
    # Scatter plot with different colors for each cluster
    plt.scatter(data[:, 0], data[:, 1], c=labels.ravel(), cmap='viridis', marker='o', label='Data Points')
    plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='^', s=100, label='Initial Centroids')
    plt.scatter(new_centroids[:, 0], new_centroids[:, 1], color='green', marker='x', s=100, label='New Centroids')
    
    plt.title('K-Means Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    
    # Use plt.show() and block the execution
    plt.show()

if __name__ == '__main__':
    main()