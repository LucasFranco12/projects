import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import pygame
import os
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def predict(x, w1, b1, W2, b2):
    z1 = np.dot(x, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))  # Subtract max for numerical stability
    a2 = a2 / np.sum(a2, axis=1, keepdims=True)
    return a2

def model(x_train, y_train, x_val, y_val, size=128, learning_rate=0.01, iterations=1000):
    input_size = x_train.shape[1]
    output_size = 10

    # Initialize weights and biases
    np.random.seed(0)
    w1 = np.random.randn(input_size, size) * 0.01
    b1 = np.zeros((1, size))
    W2 = np.random.randn(size, output_size) * 0.01
    b2 = np.zeros((1, output_size))

    epsilon = 1e-8  # Small value to prevent log(0)

    for i in range(iterations):
        # Forward propagation
        z1 = np.dot(x_train, w1) + b1
        a1 = relu(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))  # Subtract max for numerical stability
        a2 = a2 / np.sum(a2, axis=1, keepdims=True)

        # Loss function
        log_prob = -np.log(a2[range(len(y_train)), np.argmax(y_train, axis=1)] + epsilon)
        loss = np.sum(log_prob) / len(y_train)

        # Backward propagation
        dz2 = a2 - y_train
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * relu_derivative(z1)
        dW1 = np.dot(x_train.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Gradient clipping
        dW2 = np.clip(dW2, -1, 1)
        db2 = np.clip(db2, -1, 1)
        dW1 = np.clip(dW1, -1, 1)
        db1 = np.clip(db1, -1, 1)

        # Update weights and biases
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        w1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        # Print loss every 100 iterations
        if (i + 1) % 100 == 0:
            print(f"Iteration {i + 1}, Loss: {loss}")

    y_pred_val = predict(x_val, w1, b1, W2, b2)
    val_accuracy = np.mean(np.argmax(y_pred_val, axis=1) == np.argmax(y_val, axis=1)) * 100

    print(f"Validation Accuracy: {val_accuracy}%")

    return w1, b1, W2, b2, val_accuracy

def load_model():
    if os.path.exists('model_weights.npz'):
        data = np.load('model_weights.npz')
        return data['w1'], data['b1'], data['W2'], data['b2']
    else:
        return None, None, None, None

def draw_and_predict(w1, b1, W2, b2):
    pygame.init()
    screen = pygame.display.set_mode((280, 280))
    pygame.display.set_caption("Draw a digit")
    clock = pygame.time.Clock()
    canvas = np.zeros((28, 28))

    drawing = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    x = canvas.reshape(1, -1)
                    x = x / 255.0  # Normalize the input
                    x = gaussian_filter(x, sigma=1)  # Apply Gaussian blur
                    prediction = np.argmax(predict(x, w1, b1, W2, b2))
                    print(f"Predicted Digit: {prediction}")
                elif event.key == pygame.K_c:
                    canvas.fill(0)
                    screen.fill((0, 0, 0))

        if drawing:
            x, y = pygame.mouse.get_pos()
            if 0 <= x < 280 and 0 <= y < 280:
                canvas[y // 10, x // 10] = 255  # Set pixel to white
                pygame.draw.rect(screen, (255, 255, 255), (x // 10 * 10, y // 10 * 10, 10, 10))

        pygame.display.flip()
        clock.tick(60)

def hyperparameter_tuning(x_train, y_train, x_val, y_val):
    best_accuracy = 0
    best_params = None
    best_model = None

    # Define hyperparameter grid
    sizes = [64, 128, 256]
    learning_rates = [0.01, 0.005, 0.001]
    iterations = [1000, 2000, 3000, 5000, 10000]

    for size in sizes:
        for learning_rate in learning_rates:
            for iteration in iterations:
                print(f"Training with size={size}, learning_rate={learning_rate}, iterations={iteration}")
                w1, b1, W2, b2, val_accuracy = model(x_train, y_train, x_val, y_val, size=size, learning_rate=learning_rate, iterations=iteration)
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    best_params = (size, learning_rate, iteration)
                    best_model = (w1, b1, W2, b2)

    print(f"Best Validation Accuracy: {best_accuracy}% with params: size={best_params[0]}, learning_rate={best_params[1]}, iterations={best_params[2]}")
    return best_model

def main():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

    # One-hot encode the labels
    num_classes = 10
    y_train_one_hot = np.zeros((y_train.shape[0], num_classes))
    y_train_one_hot[np.arange(y_train.shape[0]), y_train] = 1
    y_test_one_hot = np.zeros((y_test.shape[0], num_classes))
    y_test_one_hot[np.arange(y_test.shape[0]), y_test] = 1

    # Split training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train_one_hot, test_size=0.2, random_state=42)

    # Perform hyperparameter tuning
    w1, b1, W2, b2 = hyperparameter_tuning(x_train, y_train, x_val, y_val)

    # Evaluate on test set
    y_pred_test = predict(x_test, w1, b1, W2, b2)
    test_accuracy = np.mean(np.argmax(y_pred_test, axis=1) == y_test) * 100
    print(f"Test Accuracy: {test_accuracy}%")
    np.savez('model_weights.npz', w1=w1, b1=b1, W2=W2, b2=b2)

    draw_and_predict(w1, b1, W2, b2)

if __name__ == '__main__':
    main()