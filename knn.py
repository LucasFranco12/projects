import numpy as np
import matplotlib.pyplot as plt


def knnClassifier(X, y, targetPoints, k=5):
    targetClassifier = []
    for t in targetPoints:
        distance = []
        for i in range(len(X)):
            euclidean_distance = np.sqrt(np.sum((X[i] - t) ** 2))
            distance.append((euclidean_distance, y[i]))
        distance.sort(key=lambda x: x[0])
        distance = distance[:k]
        group1 = 0
        group2 = 0
        for d in distance:
            if d[1] == 0:
                group1 +=1
            elif d[1] == 1:
                group2 +=1
        if group1 > group2:
            targetClassifier.append(0)
        else:
            targetClassifier.append(1)

    return targetClassifier


def main():

    np.random.seed(42)
    X1 = np.random.normal(loc=[2, 2], scale=1, size=(50, 2))
    X2 = np.random.normal(loc=[4, 4], scale=1, size=(50, 2))
    X = np.vstack((X1, X2)) # splits first 50 points around location 2,2 and next 50 points around 4,4
    y = np.hstack((np.zeros(50), np.ones(50))) # classifies first 50 points as 0 and next 50 points as 1
    targetPoints = [[3, 3], [2, 3], [4, 5], [5, 5], [3, 4], [4, 3]]


    targetIdentifier = knnClassifier(X, y, targetPoints)

    plt.figure(figsize=(8, 6))

    plt.scatter(X[:50,0],X[:50,1] , color='red',  label='Group 1')
    plt.scatter(X[50:, 0], X[50:, 1], color='blue', label='Group 2')


    for i in range(len(targetPoints)):
        if targetIdentifier[i] == 0:
            plt.scatter(targetPoints[i][0], targetPoints[i][1], color='red', marker='x', s=100, label='Target Point')
        else:
            plt.scatter(targetPoints[i][0], targetPoints[i][1], color='blue', marker='x', s=100, label='Target Point')
    plt.title('Two Groups Scatter Plot')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
   # plt.legend()
    plt.show()

if __name__ == "__main__":
    main()