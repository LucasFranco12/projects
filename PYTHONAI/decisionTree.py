import numpy as np
import pygame
import sys

def create_weather_dataset(n_samples):
    """
    Creates a fake weather dataset 
    
    Features:
    - Temperature (°C)
    - Humidity (%)
    - Wind Speed (km/h)
    - Pressure (hPa - normalized)
    
    Target:
    - Rain (0: No Rain, 1: Rain)
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate feature data
    temperature = np.random.uniform(0, 35, n_samples)  # Temperature in Celsius
    humidity = np.random.uniform(30, 95, n_samples)    # Humidity percentage
    wind_speed = np.random.uniform(0, 50, n_samples)   # Wind speed in km/h
    pressure = np.random.uniform(-1, 1, n_samples)     # Normalized pressure
    
    # Create feature matrix
    X = np.column_stack([temperature, humidity, wind_speed, pressure])
    
    # Generate target variable (rain/no rain) based on simple rules
    y = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Higher chance of rain if:
        # - Temperature is moderate (10-20°C)
        # - Humidity is high (>70%)
        # - Pressure is low (<0 in our normalized scale)
        rain_score = 0
        
        if 10 <= temperature[i] <= 20:
            rain_score += 1
        if humidity[i] > 70:
            rain_score += 1
        if pressure[i] < 0:
            rain_score += 1
            
        # Assign rain (1) if enough conditions are met
        if rain_score >= 2:
            y[i] = 1
    
    return X, y

def gini_impurity(y):
    # Calculate the Gini Impurity for a list of labels.
    m = len(y)
    if m == 0:
        return 0
    p = np.sum(y) / m  # p is porportion of positive class 
    return 1 - p**2 - (1 - p)**2

def split_dataset(X, y, feature_index, threshold):
    # Split the dataset based on a feature and threshold.
    left_mask = X[:, feature_index] <= threshold
    right_mask = X[:, feature_index] > threshold
    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

def best_split(X, y):
    # Find the best feature and threshold to split the dataset.
    m, n = X.shape
    best_gini = float('inf')
    best_feature_index = None
    best_threshold = None
    for feature_index in range(n):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            X_left, X_right, y_left, y_right = split_dataset(X, y, feature_index, threshold)
            gini_left = gini_impurity(y_left)
            gini_right = gini_impurity(y_right)
            gini = (len(y_left) * gini_left + len(y_right) * gini_right) / m
            if gini < best_gini:
                best_gini = gini
                best_feature_index = feature_index
                best_threshold = threshold
    return best_feature_index, best_threshold

def build_tree(X, y, max_depth=None, min_samples_split=2, depth=0):
    # Build the decision tree recursively 
    if len(set(y)) == 1 or len(y) < min_samples_split or (max_depth is not None and depth >= max_depth):
        return {
            'gini': gini_impurity(y),
            'num_samples': len(y),
            'num_samples_per_class': [np.sum(y == i) for i in np.unique(y)],
            'predicted_class': max(set(y), key=list(y).count),
            'feature_index': None,
            'threshold': None,
            'left': None,
            'right': None
        }
    
    num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
    predicted_class = np.argmax(num_samples_per_class)
    node = {
        'gini': gini_impurity(y),
        'num_samples': len(y),
        'num_samples_per_class': num_samples_per_class,
        'predicted_class': predicted_class,
        'feature_index': None,
        'threshold': None,
        'left': None,
        'right': None
    }

    feature_index, threshold = best_split(X, y)
    if feature_index is None:
        return node

    node['feature_index'] = feature_index
    node['threshold'] = threshold
    X_left, X_right, y_left, y_right = split_dataset(X, y, feature_index, threshold)
    node['left'] = build_tree(X_left, y_left, max_depth, min_samples_split, depth + 1)
    node['right'] = build_tree(X_right, y_right, max_depth, min_samples_split, depth + 1)
    return node

def predict_tree(node, X):
   # Predict the class for a sample using the decision tree.
    if node['feature_index'] is None:
        return node['predicted_class']
    if X[node['feature_index']] <= node['threshold']:
        return predict_tree(node['left'], X)
    else:
        return predict_tree(node['right'], X)

def draw_tree(screen, node, x, y, dx, dy, feature_names, depth=0):
    # Draw tree recursively with Pygame
    if node['feature_index'] is None:
        color = (0, 255, 0) if node['predicted_class'] == 1 else (255, 0, 0)
        pygame.draw.circle(screen, color, (x, y), 20)
        font = pygame.font.SysFont(None, 24)
        text = font.render(str(node['predicted_class']), True, (0, 0, 0))
        screen.blit(text, (x - 10, y - 10))
    else:
        feature_name = feature_names[node['feature_index']]
        threshold = node['threshold']
        font = pygame.font.SysFont(None, 24)
        text = font.render(f"{feature_name} <= {threshold:.2f}", True, (0, 0, 0))
        screen.blit(text, (x - 50, y - 30))
        textss = font.render("True is left False is right", True, (0, 0, 0))
        screen.blit(textss, (0, 0))
        pygame.draw.circle(screen, (0, 0, 255), (x, y), 20)
        pygame.draw.line(screen, (0, 0, 0), (x, y), (x - dx, y + dy), 2)
        pygame.draw.line(screen, (0, 0, 0), (x, y), (x + dx, y + dy), 2)
        draw_tree(screen, node['left'], x - dx, y + dy, dx // 2, dy, feature_names, depth + 1)
        draw_tree(screen, node['right'], x + dx, y + dy, dx // 2, dy, feature_names, depth + 1)

def main():
    # Create dataset
    X, y = create_weather_dataset(5000)
    feature_names = ["Temperature", "Humidity", "Wind Speed", "Pressure"]

    # Build decision tree
    tree = build_tree(X, y, max_depth=5, min_samples_split=10)
    
    # Predict on a sample
    sample = X[2]
    prediction = predict_tree(tree, sample)
    print(f"Predicted class for sample {sample}: {prediction} vs. true class: {y[2]}")

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption("Decision Tree Visualization")
    screen.fill((255, 255, 255))

    # Draw the decision tree
    draw_tree(screen, tree, 600, 50, 300, 100, feature_names)

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()