import numpy as np
import matplotlib.pyplot as plt



# Logistic regression functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(weights, bias, X):
    return sigmoid(np.dot(X, weights) + bias)

def compute_cost(weights, bias, X, y):
    m = len(y)
    predictions = predict(weights, bias, X)
    epsilon = 1e-5  # Small value to prevent log(0)
    cost = -(1/m) * np.sum(y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon))
    return cost

def logisitcRegression(weights, bias, X, y, learning_rate, num_iterations):
   
    # Gradient Descent
    m = len(y)
    for i in range(num_iterations):
        predictions = predict(weights, bias, X)
        dw = (1/m) * np.dot(X.T, (predictions - y))
        db = (1/m) * np.sum(predictions - y)
        weights -= learning_rate * dw
        bias -= learning_rate * db
        if i % 100 == 0:
            predictions = predict(weights, bias, X)
            epsilon = 1e-5  # Small value to prevent log(0)
            cost = -(1/m) * np.sum(y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon))

            print(f"Iteration {i}: Cost {cost}")
    return weights, bias

def main():
    # Random fake data
    np.random.seed(0)
    num_samples = 100
    weights_data = np.random.normal(180, 30, num_samples)  # Mean weight 180 lbs, std dev 30 lbs

    obesity = (weights_data > 200).astype(int)  # 1 if weight > 200 lbs, else 0

    # Normalize the weights data
    weights_data_normalized = (weights_data - np.mean(weights_data)) / np.std(weights_data)
    # Prepare data
    X = weights_data_normalized.reshape(-1, 1)
    y = obesity

    # Initialize parameters
    initial_weights = np.zeros(X.shape[1])
    initial_bias = 0
    learning_rate = 0.01
    num_iterations = 10000

    # Train logistic regression model
    trained_weights, trained_bias = logisitcRegression(initial_weights, initial_bias, X, y, learning_rate, num_iterations)

    # Plot results
    plt.scatter(weights_data, y, color='red', label='Data points')
    x_values = np.linspace(min(weights_data), max(weights_data), 100)
    x_values_normalized = (x_values - np.mean(weights_data)) / np.std(weights_data)
    y_values = predict(trained_weights, trained_bias, x_values_normalized.reshape(-1, 1))
    plt.plot(x_values, y_values, color='blue', label='Logistic Regression')
    plt.xlabel('Weight (lbs)')
    plt.ylabel('Probability of Obesity')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()