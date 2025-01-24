import cv2
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class ChessPieceDetector:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_training_data(self, stats_list, labels):
        """Convert stats dictionaries to feature matrix"""
        features = []
        for stats in stats_list:
            feature_vector = [
                stats['hsv_hue_mean'],
                stats['hsv_sat_mean'],
                stats['hsv_val_mean'],
                stats['hsv_hue_std'],
                stats['hsv_sat_std'],
                stats['hsv_val_std'],
                stats['gray_mean'],
                stats['gray_std']
            ]
            features.append(feature_vector)
        return np.array(features)
        
    def train(self, training_stats, labels):
        """Train the classifier on labeled data"""
        X = self.prepare_training_data(training_stats, labels)
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, labels)
        self.is_trained = True
        
    def predict(self, stats):
        """Predict if a square contains a piece"""
        if not self.is_trained:
            raise RuntimeError("Detector needs to be trained first")
            
        features = self.prepare_training_data([stats], None)[0].reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        return self.classifier.predict(features_scaled)[0]

class ChessVision:
    def __init__(self):
        self.points = []
        self.last_board_state = self.get_initial_board_state()
        self.h_lines = None
        self.v_lines = None
        self.last_frame = None
        self.selected_square = None
        self.piece_detector = ChessPieceDetector()
        self.training_data = []
        self.is_collecting_data = False  # Start in data collection mode
        self.is_calibrating = True
        self.calibration_complete = False

    def start_calibration(self):
        """Reset calibration points"""
        self.points = []
        self.is_calibrating = True
        self.calibration_complete = False
        self.h_lines = None
        self.v_lines = None
        
    def add_calibration_point(self, x, y):
        """Add a calibration point and check if calibration is complete"""
        if len(self.points) < 4:
            self.points.append((x, y))
            
        if len(self.points) == 4:
            self.calibration_complete = True
            self.is_calibrating = False
            self.is_collecting_data = True  # Start collecting data after calibration
            # Reset lines to force recalculation
            self.h_lines = None
            self.v_lines = None
            return True
        return False

    def draw_calibration(self, frame):
        """Draw calibration points and lines"""
        # Draw points
        for i, point in enumerate(self.points):
            cv2.circle(frame, point, 5, (0, 0, 255), -1)
            cv2.putText(frame, str(i+1), (point[0]+10, point[1]+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        # Draw lines between points
        if len(self.points) > 1:
            for i in range(len(self.points)):
                if i < len(self.points) - 1:
                    cv2.line(frame, self.points[i], self.points[i+1], (0, 255, 0), 2)
                elif len(self.points) == 4:  # Complete the rectangle
                    cv2.line(frame, self.points[i], self.points[0], (0, 255, 0), 2)
        
        # Draw instructions
        if not self.calibration_complete:
            msg = f"Click the {len(self.points)+1}th corner of the board"
            cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2)


    def get_initial_board_state(self):
        """Initialize 8x8 board with starting position"""
        board = np.zeros((8, 8), dtype=int)
        board[0:2, :] = 1  # Black pieces
        board[6:8, :] = 1  # White pieces
        return board
    
    def get_square_points(self):
        """Calculate grid points with perspective correction"""
        if self.h_lines is not None and self.v_lines is not None:
            return self.h_lines, self.v_lines
            
        pts = np.float32(self.points)
        h_lines = []
        v_lines = []
        
        for i in range(9):
            left = pts[0] + (pts[3] - pts[0]) * i / 8.0
            right = pts[1] + (pts[2] - pts[1]) * i / 8.0
            h_line = [left + (right - left) * j / 8.0 for j in range(9)]
            h_lines.append(h_line)
            
            top = pts[0] + (pts[1] - pts[0]) * i / 8.0
            bottom = pts[3] + (pts[2] - pts[3]) * i / 8.0
            v_line = [top + (bottom - top) * j / 8.0 for j in range(9)]
            v_lines.append(v_line)
            
        self.h_lines = h_lines
        self.v_lines = v_lines
        return h_lines, v_lines
    
    def analyze_square(self, frame, square_corners, row, col):
        """Analyze square and return detailed color statistics"""
        corners = [tuple(map(int, corner)) for corner in square_corners]
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(corners)], 255)
        
        roi = cv2.bitwise_and(frame, frame, mask=mask)
        
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        hsv_mean = cv2.mean(hsv_roi, mask=mask)
        hsv_std = cv2.meanStdDev(hsv_roi, mask=mask)[1]
        gray_mean = cv2.mean(gray_roi, mask=mask)[0]
        gray_std = cv2.meanStdDev(gray_roi, mask=mask)[1][0][0]
        
        stats = {
            'pos': f"{row},{col}",
            'hsv_hue_mean': hsv_mean[0],
            'hsv_sat_mean': hsv_mean[1],
            'hsv_val_mean': hsv_mean[2],
            'hsv_hue_std': hsv_std[0][0],
            'hsv_sat_std': hsv_std[1][0],
            'hsv_val_std': hsv_std[2][0],
            'gray_mean': gray_mean,
            'gray_std': gray_std
        }
        
        return stats
    
    def is_empty_square(self, stats):
        """Detect if square is empty using trained classifier"""
        if not self.piece_detector.is_trained:
            # Default to simple threshold-based detection during data collection
            return (stats['hsv_val_mean'] > 180 and 
                    stats['hsv_sat_mean'] < 20 and 
                    stats['hsv_val_std'] < 35)
        return self.piece_detector.predict(stats)
    
    def train_detector(self):
        """Train the piece detector using collected data"""
        if not self.training_data:
            print("No training data available")
            return
            
        training_stats = []
        labels = []
        
        for stats in self.training_data:
            row = int(stats['pos'].split(',')[0])
            # Label squares in rows 2-5 as empty (1), others as occupied (0)
            label = 1 if 2 <= row <= 5 else 0
            training_stats.append(stats)
            labels.append(label)
        
        self.piece_detector.train(training_stats, labels)
        print("Detector trained on", len(training_stats), "squares")
        self.is_collecting_data = False  # Switch to detection mode
    
    def process_frame(self, frame):
        h_lines, v_lines = self.get_square_points()
        square_stats = []
        
        # Process each square
        for i in range(8):
            for j in range(8):
                corners = [
                    h_lines[i][j],
                    h_lines[i][j+1],
                    h_lines[i+1][j+1],
                    h_lines[i+1][j]
                ]
                
                stats = self.analyze_square(frame, corners, i, j)
                square_stats.append(stats)
                
                # Collect training data if in collection mode
                if self.is_collecting_data:
                    self.training_data.append(stats)
                
                # Draw grid
                for k in range(4):
                    pt1 = tuple(map(int, corners[k]))
                    pt2 = tuple(map(int, corners[(k+1)%4]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
                
                # Draw square position and status
                center = tuple(map(int, sum(corners)/4))
                is_empty = self.is_empty_square(stats)
                color = (0, 255, 0) if is_empty else (0, 0, 255)
                
                cv2.putText(frame, f"{i},{j}", center, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Highlight detected pieces
                if not is_empty:
                    cv2.circle(frame, center, 5, color, -1)

                # Highlight selected square if any
                if self.selected_square == (i, j):
                    cv2.circle(frame, center, 8, (255, 0, 0), -1)
        
        return frame, square_stats

def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks for both calibration and square selection"""
    chess_vision = param
    if event == cv2.EVENT_LBUTTONDOWN:
        if chess_vision.is_calibrating:
            if chess_vision.add_calibration_point(x, y):
                print("Calibration complete! Now collecting training data...")
                print("Press 'T' when ready to train the detector")
        elif chess_vision.calibration_complete:
            h_lines, v_lines = chess_vision.get_square_points()
            for i in range(8):
                for j in range(8):
                    corners = [
                        h_lines[i][j],
                        h_lines[i][j+1],
                        h_lines[i+1][j+1],
                        h_lines[i+1][j]
                    ]
                    corners = np.array(corners, dtype=np.int32)
                    if cv2.pointPolygonTest(corners, (x, y), False) >= 0:
                        chess_vision.selected_square = (i, j)
                        return

def test_webcam():
    cap = cv2.VideoCapture(0)
    chess_vision = ChessVision()
    
    cv2.namedWindow('Chess Vision')
    cv2.setMouseCallback('Chess Vision', mouse_callback, chess_vision)
    
    print("Chess Vision Analysis started.")
    print("Please calibrate the board first:")
    print("1. Click the top-left corner")
    print("2. Click the top-right corner")
    print("3. Click the bottom-right corner")
    print("4. Click the bottom-left corner")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if chess_vision.is_calibrating:
            # Just show frame with calibration points
            chess_vision.draw_calibration(frame)
            cv2.imshow('Chess Vision', frame)
        else:
            # Normal processing
            processed_frame, stats = chess_vision.process_frame(frame)
            cv2.imshow('Chess Vision', processed_frame)
            
            # Handle selected square analysis
            if chess_vision.selected_square is not None:
                for stat in stats:
                    if stat['pos'] == f"{chess_vision.selected_square[0]},{chess_vision.selected_square[1]}":
                        print("\nSquare Analysis:")
                        print(f"Position: {stat['pos']}")
                        print(f"HSV Means - H: {stat['hsv_hue_mean']:.1f}, S: {stat['hsv_sat_mean']:.1f}, V: {stat['hsv_val_mean']:.1f}")
                        print(f"HSV StdDev - H: {stat['hsv_hue_std']:.1f}, S: {stat['hsv_sat_std']:.1f}, V: {stat['hsv_val_std']:.1f}")
                        print(f"Grayscale - Mean: {stat['gray_mean']:.1f}, StdDev: {stat['gray_std']:.1f}")
                        chess_vision.selected_square = None
                        break
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("Restarting calibration...")
            chess_vision.start_calibration()
        elif key == ord('t') and chess_vision.is_collecting_data:
            print("Training detector...")
            chess_vision.train_detector()
            print("Detector trained! Now using ML-based detection")
        
        # Collect training data every N frames after calibration
        if chess_vision.is_collecting_data and not chess_vision.piece_detector.is_trained:
            frame_count += 1
            if frame_count >= 30:
                frame_count = 0
                print("Collected data from frame")
    
    cap.release()
    cv2.destroyAllWindows()

# ... [ChessPieceDetector class remains the same] ...

if __name__ == "__main__":
    test_webcam()