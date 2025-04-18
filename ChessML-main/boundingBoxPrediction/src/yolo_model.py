import torch
from pathlib import Path
import yaml
import shutil
import os
import json
from ultralytics import YOLO
from typing import Optional # Import Optional

class ChessPieceDetector:
    def __init__(self, 
                 data_yaml_path: Optional[str] = None, 
                 annotations_path: Optional[str] = None, 
                 weights_path: Optional[str] = None):
        
        self.data_yaml = Path(data_yaml_path) if data_yaml_path else None
        self.annotations_path = Path(annotations_path) if annotations_path else None
        self.model = None
        
 
        if weights_path:
            print(f"Loading model from weights: {weights_path}")
            try:
                self.model = YOLO(weights_path)
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error loading model weights: {e}")
                raise

        self.class_mapping = {
            'white_pawn': 0, 'white_rook': 1, 'white_knight': 2,
            'white_bishop': 3, 'white_queen': 4, 'white_king': 5,
            'black_pawn': 6, 'black_rook': 7, 'black_knight': 8,
            'black_bishop': 9, 'black_queen': 10, 'black_king': 11,
            'empty_square': 12
        }

    
        self.class_mapping_predict = {
            0: 'white_pawn', 1: 'white_rook', 2: 'white_knight', 
            3: 'white_bishop', 4: 'white_queen', 5: 'white_king', 
            6: 'black_pawn', 7: 'black_rook', 8: 'black_knight', 
            9: 'black_bishop', 10: 'black_queen', 11: 'black_king'
            # 12: 'empty_square' # Exclude empty for bbox prediction
        }

    def train(self, epochs: int = 100, batch: int = 16, imgsz: int = 800):
        """Train the YOLO model"""

        if not self.data_yaml:
            raise ValueError("data_yaml_path must be provided for training.")
            
        try:

            if self.model is None:
                print("Initializing model with yolov8n.pt for training...")
                self.model = YOLO('yolov8n.pt') 
            else:
                print("Continuing training with pre-loaded weights...")

            print("\nStarting YOLO training...")
            results = self.model.train(
                data=str(self.data_yaml),
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                device='0' if torch.cuda.is_available() else 'cpu'
            )
            

            print("Training complete.")
            return results
            
        except Exception as e:
            print(f"Error during training: {e}")
            raise

    def predict(self, image_path: str, confidence_threshold: float = 0.25):
        """Perform prediction on a single image."""
        if self.model is None:
            raise ValueError("Model not loaded. Initialize with weights_path or train first.")
            
        try:
            print(f"Running prediction on: {image_path}")
    
            results = self.model(image_path, conf=confidence_threshold)
            print("Prediction complete.")
            return results
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise

    def get_predictions_details(self, results) -> list:
        """Extracts and formats prediction details from YOLO results."""
        details = []

        if not results or len(results) == 0:
            return details

        result = results[0] 
        boxes = result.boxes 

        for i in range(len(boxes)):
            box = boxes[i]
            class_id = int(box.cls.item()) # Get class ID
            confidence = float(box.conf.item()) # Get confidence score
            bbox_coords = box.xyxy.tolist()[0] # Get bounding box coordinates [x1, y1, x2, y2]
            
            class_name = self.class_mapping_predict.get(class_id, f"Unknown_ID_{class_id}")
            
            details.append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': bbox_coords # [x_min, y_min, x_max, y_max]
            })
            
        return details