import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class BiometricPredictor:
    def __init__(self):
        self.features = ['calories', 'protein', 'carbs', 'fats']
        self.weight_model = None
        self.bodyfat_model = None
    
    def prepare_data(self, data):
        # Same as before
        X = np.array([[record[feature] for feature in self.features] for record in data])
        weights = np.array([record['weight'] for record in data])
        bodyfats = np.array([record['bodyfat'] for record in data])
        
        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0)
        X_normalized = (X - self.feature_means) / self.feature_stds
        
        return X_normalized, weights, bodyfats
    
    def normalize_input(self, input_data):
        if isinstance(input_data, list):
            X = np.array([[record[feature] for feature in self.features] for record in input_data])
        else:
            X = np.array([[input_data[feature] for feature in self.features]])
        return (X - self.feature_means) / self.feature_stds
    
    def analyze_feature_importance(self):
        """
        Analyze the importance of features in predicting weight and body fat.
        
        Returns:
        dict: A dictionary containing feature importance for weight and body fat predictions
        """
        if self.weight_model is None or self.bodyfat_model is None:
            raise ValueError("Models must be trained before analyzing feature importance")
        
        # Exclude the bias term (first coefficient)
        weight_coefficients = self.weight_model[1:]
        bodyfat_coefficients = self.bodyfat_model[1:]
        
        # Calculate absolute importance
        weight_importance = np.abs(weight_coefficients)
        bodyfat_importance = np.abs(bodyfat_coefficients)
        
        # Normalize to percentages
        weight_importance_normalized = weight_importance / np.sum(weight_importance) * 100
        bodyfat_importance_normalized = bodyfat_importance / np.sum(bodyfat_importance) * 100
        
        # Create result dictionary
        importance = {
            'weight': dict(zip(self.features, weight_importance_normalized)),
            'bodyfat': dict(zip(self.features, bodyfat_importance_normalized))
        }
        
        return importance

    def plot_feature_importance(self, importance=None):
        """
        Plot feature importance for weight and body fat predictions.
        
        Args:
        importance (dict, optional): Precomputed feature importance. 
                                    If None, will compute it.
        """
        import matplotlib.pyplot as plt
        
        if importance is None:
            importance = self.analyze_feature_importance()
        
        plt.figure(figsize=(10, 5))
        
        # Weight importance subplot
        plt.subplot(1, 2, 1)
        plt.bar(importance['weight'].keys(), importance['weight'].values())
        plt.title('Feature Importance for Weight Prediction')
        plt.ylabel('Importance (%)')
        plt.xticks(rotation=45)
        
        # Body Fat importance subplot
        plt.subplot(1, 2, 2)
        plt.bar(importance['bodyfat'].keys(), importance['bodyfat'].values())
        plt.title('Feature Importance for Body Fat Prediction')
        plt.ylabel('Importance (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()


    def plot_regression(self, training_data):
        """
        Create visualization of regression predictions for weight and body fat.
        
        Args:
        training_data (list): The original training dataset
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Prepare data for plotting
        X = np.array([[record[feature] for feature in self.features] for record in training_data])
        actual_weights = np.array([record['weight'] for record in training_data])
        actual_bodyfats = np.array([record['bodyfat'] for record in training_data])
        
        # Normalize input
        X_normalized = self.normalize_input(training_data)
        X_b = np.c_[np.ones((X_normalized.shape[0], 1)), X_normalized]
        
        # Predict using the trained models
        predicted_weights = X_b.dot(self.weight_model)
        predicted_bodyfats = X_b.dot(self.bodyfat_model)

        # Create two subplots
        plt.figure(figsize=(12, 5))

        # Weight Prediction Plot
        plt.subplot(1, 2, 1)
        plt.scatter(actual_weights, predicted_weights, color='blue', label='Predictions')
        plt.plot([actual_weights.min(), actual_weights.max()], 
                [actual_weights.min(), actual_weights.max()], 
                color='red', linestyle='--', label='Ideal Prediction')
        plt.title('Weight: Actual vs Predicted')
        plt.xlabel('Actual Weight (lbs)')
        plt.ylabel('Predicted Weight (lbs)')
        plt.legend()

        # Body Fat Prediction Plot
        plt.subplot(1, 2, 2)
        plt.scatter(actual_bodyfats, predicted_bodyfats, color='green', label='Predictions')
        plt.plot([actual_bodyfats.min(), actual_bodyfats.max()], 
                [actual_bodyfats.min(), actual_bodyfats.max()], 
                color='red', linestyle='--', label='Ideal Prediction')
        plt.title('Body Fat: Actual vs Predicted')
        plt.xlabel('Actual Body Fat (%)')
        plt.ylabel('Predicted Body Fat (%)')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def train(self, data):
        if len(data) < 5:
            print("Warning: Model accuracy may be low with fewer than 5 data points.")
            
        X, weights, bodyfats = self.prepare_data(data)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Add small regularization term
        reg_term = 1e-8 * np.eye(X_b.shape[1])
        
        try:
            # Train weight model
            self.weight_model = np.linalg.pinv(X_b.T.dot(X_b) + reg_term).dot(X_b.T).dot(weights)
            
            # Train bodyfat model
            self.bodyfat_model = np.linalg.pinv(X_b.T.dot(X_b) + reg_term).dot(X_b.T).dot(bodyfats)
            
            # Calculate metrics
            weight_predictions = self.predict_weight(data)
            bodyfat_predictions = self.predict_bodyfat(data)
            
            weight_mse = np.mean((weight_predictions - weights) ** 2)
            bodyfat_mse = np.mean((bodyfat_predictions - bodyfats) ** 2)
            
            return {
                'weight_mse': weight_mse,
                'weight_rmse': np.sqrt(weight_mse),
                'bodyfat_mse': bodyfat_mse,
                'bodyfat_rmse': np.sqrt(bodyfat_mse)
            }
        
        except np.linalg.LinAlgError as e:
            print("Error during training: ", e)
            print("Try adding more diverse data points.")
            return None
    
    def predict_weight(self, input_data):
        X = self.normalize_input(input_data)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.weight_model)
    
    def predict_bodyfat(self, input_data):
        X = self.normalize_input(input_data)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.bodyfat_model)
    
    # Rest of the methods remain the same...

# Main execution
if __name__ == "__main__":
    # Sample training data with more points
    training_data = [
        {
            'date': '2024-01-01',
            'calories': 2600,
            'protein': 204,
            'carbs': 193,
            'fats': 104,
            'weight': 178,
            'bodyfat': 15
        },
        {
            'date': '2024-01-02',
            'calories': 2500,
            'protein': 200,
            'carbs': 220,
            'fats': 91,
            'weight': 179,
            'bodyfat': 14.6
        },
        {
            'date': '2024-01-03',
            'calories': 1800,
            'protein': 140,
            'carbs': 180,
            'fats': 55,
            'weight': 179,
            'bodyfat': 14.8
        },
        {
            'date': '2024-01-04',
            'calories': 2300,
            'protein': 170,
            'carbs': 230,
            'fats': 70,
            'weight': 181.5,
            'bodyfat': 15.3
        },
        {
            'date': '2024-01-05',
            'calories': 1900,
            'protein': 145,
            'carbs': 190,
            'fats': 58,
            'weight': 180,
            'bodyfat': 14.9
        }
        ,
     {
        "date": "2024-01-01",
        "calories": 2600,
        "protein": 180,
        "carbs": 300,
        "fats": 65,
        "weight": 180.7,
        "bodyfat": 14.6
    },
    {
        "date": "2024-01-08",
        "calories": 2550,
        "protein": 180,
        "carbs": 290,
        "fats": 65,
        "weight": 179.3,
        "bodyfat": 14.3
    },
    {
        "date": "2024-01-15",
        "calories": 2500,
        "protein": 180,
        "carbs": 280,
        "fats": 65,
        "weight": 180.7,
        "bodyfat": 15.4
    },
    {
        "date": "2024-01-22",
        "calories": 2450,
        "protein": 180,
        "carbs": 270,
        "fats": 65,
        "weight": 180.6,
        "bodyfat": 15.0
    },
    {
        "date": "2024-02-01",
        "calories": 2550,
        "protein": 180,
        "carbs": 260,
        "fats": 65,
        "weight": 179.6,
        "bodyfat": 14.6
    },
    {
        "date": "2024-02-08",
        "calories": 2500,
        "protein": 180,
        "carbs": 250,
        "fats": 65,
        "weight": 179.8,
        "bodyfat": 15.7
    },
    {
        "date": "2024-02-15",
        "calories": 2450,
        "protein": 180,
        "carbs": 240,
        "fats": 65,
        "weight": 179.9,
        "bodyfat": 13.9
    },
    {
        "date": "2024-02-22",
        "calories": 2400,
        "protein": 180,
        "carbs": 230,
        "fats": 65,
        "weight": 179.7,
        "bodyfat": 16.0
    },
    {
        "date": "2024-03-01",
        "calories": 2500,
        "protein": 180,
        "carbs": 220,
        "fats": 65,
        "weight": 180.8,
        "bodyfat": 14.5
    },
    {
        "date": "2024-03-08",
        "calories": 2450,
        "protein": 180,
        "carbs": 210,
        "fats": 65,
        "weight": 180.0,
        "bodyfat": 15.7
    },
    {
        "date": "2024-03-15",
        "calories": 2400,
        "protein": 180,
        "carbs": 200,
        "fats": 65,
        "weight": 179.6,
        "bodyfat": 15.9
    },
    {
        "date": "2024-03-22",
        "calories": 2350,
        "protein": 180,
        "carbs": 190,
        "fats": 65,
        "weight": 180.3,
        "bodyfat": 14.8
    },
    {
        "date": "2024-04-01",
        "calories": 2450,
        "protein": 180,
        "carbs": 180,
        "fats": 65,
        "weight": 180.5,
        "bodyfat": 14.9
    },
    {
        "date": "2024-04-08",
        "calories": 2400,
        "protein": 180,
        "carbs": 170,
        "fats": 65,
        "weight": 179.3,
        "bodyfat": 16.1
    },
    {
        "date": "2024-04-15",
        "calories": 2350,
        "protein": 180,
        "carbs": 160,
        "fats": 65,
        "weight": 180.3,
        "bodyfat": 15.0
    },
    {
        "date": "2024-04-22",
        "calories": 2300,
        "protein": 180,
        "carbs": 150,
        "fats": 65,
        "weight": 180.5,
        "bodyfat": 15.7
    },
    {
        "date": "2024-05-01",
        "calories": 2400,
        "protein": 180,
        "carbs": 150,
        "fats": 65,
        "weight": 180.2,
        "bodyfat": 14.5
    },
    {
        "date": "2024-05-08",
        "calories": 2350,
        "protein": 180,
        "carbs": 150,
        "fats": 65,
        "weight": 179.3,
        "bodyfat": 15.5
    },
    {
        "date": "2024-05-15",
        "calories": 2300,
        "protein": 180,
        "carbs": 150,
        "fats": 65,
        "weight": 179.2,
        "bodyfat": 14.8
    },
    {
        "date": "2024-05-22",
        "calories": 2250,
        "protein": 180,
        "carbs": 150,
        "fats": 65,
        "weight": 180.8,
        "bodyfat": 16.1
    }
    ]

    # Create predictor instance
    predictor = BiometricPredictor()

    # Train the model
    print("Training the model...")
    metrics = predictor.train(training_data)
    print("Training metrics:", metrics)

    # Make a prediction for a new day
    new_day = {
        'calories': 2100,
        'protein': 155,
        'carbs': 210,
        'fats': 62
    }

    predicted_weight = predictor.predict_weight(new_day)
    predicted_bodyfat = predictor.predict_bodyfat(new_day)
    
    # Handle single prediction case
    if isinstance(predicted_weight, np.ndarray):
        predicted_weight = predicted_weight[0]
    if isinstance(predicted_bodyfat, np.ndarray):
        predicted_bodyfat = predicted_bodyfat[0]
        
    print(f"\nPredictions for new day:")
    print(f"Predicted weight: {predicted_weight:.1f} lbs")
    print(f"Predicted body fat: {predicted_bodyfat:.1f}%")

    # Analyze feature importance
    importance = predictor.analyze_feature_importance()
    print("\nFeature importance:")
    print("For weight prediction:", importance['weight'])
    print("For bodyfat prediction:", importance['bodyfat'])

    # Optionally plot the importance
    predictor.plot_feature_importance()

    # Plot the results
    print("\nGenerating plots...")
   # predictor.plot_predictions(training_data)
    # After training the model
    predictor.plot_regression(training_data)