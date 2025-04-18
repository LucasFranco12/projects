import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import time
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import shuffle

class ChessPieceDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.training_data = []
        self.labels = []
        self.is_trained = False
        self.points = []
        self.corners = []
        self.h_lines = None
        self.v_lines = None
        self.is_calibrating = True
        self.calibration_complete = False
        self.selected_point = None

    def collect_training_data(self, frame, label):
        h_lines, v_lines, corners = self.get_square_points()
        
        # Add rotation variations
        angles = [0, 5, -5, 10, -10]
        for angle in angles:
            # Rotate both frame and corner points
            rotated_frame = self._rotate_frame(frame, angle) if angle != 0 else frame
            rotated_corners = self._rotate_corners(corners, angle, frame.shape) if angle != 0 else corners
            
            for i in range(8):
                for j in range(8):
                    square_corners = rotated_corners[i * 8 + j]
                    stats = self.analyze_square(rotated_frame, square_corners)
                    self.training_data.append(stats)
                    self.labels.append(label[i][j])

    def _rotate_frame(self, frame, angle):
        """
        Rotate the frame by the specified angle around its center.
        """
        height, width = frame.shape[:2]
        center = (width/2, height/2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(frame, rotation_matrix, (width, height))

    def _rotate_corners(self, corners, angle, shape):
        """Rotate corner points to match the rotated frame"""
        height, width = shape[:2]
        center = (width/2, height/2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        rotated_corners = []
        for square_corners in corners:
            rotated_square = []
            for point in square_corners:
                x = point[0]
                y = point[1]
                new_x = rotation_matrix[0][0]*x + rotation_matrix[0][1]*y + rotation_matrix[0][2]
                new_y = rotation_matrix[1][0]*x + rotation_matrix[1][1]*y + rotation_matrix[1][2]
                rotated_square.append([new_x, new_y])
            rotated_corners.append(rotated_square)
        return rotated_corners

    def analyze_square(self, frame, square_corners):
        # Convert corners to tuples of integers
        corners = [tuple(map(int, corner)) for corner in square_corners]
        
        # Create a mask of the same size as the frame, initialized to zero
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Fill the mask with a polygon defined by the square corners
        cv2.fillPoly(mask, [np.array(corners)], 255)
        
        # Apply the mask to the frame to isolate the region of interest (ROI)
        roi = cv2.bitwise_and(frame, frame, mask=mask)

        # These create a rectangular crop
        #x, y, w, h = cv2.boundingRect(np.array(corners))
        #roi_cropped = roi[y:y+h, x:x+w]

        # Create a square crop that's large enough to contain the chess square
        corner_points = np.array(corners)
        # Find max pixel distance from points to determine square size
        max_dim = max(
            np.max(corner_points[:, 0]) - np.min(corner_points[:, 0]),  # max width - min width
            np.max(corner_points[:, 1]) - np.min(corner_points[:, 1])   # max height - min height
        )
        center_x = (np.max(corner_points[:, 0]) + np.min(corner_points[:, 0])) // 2 
        center_y = (np.max(corner_points[:, 1]) + np.min(corner_points[:, 1])) // 2
        
        # add extra area around the square
        area = max_dim // 10 
        x = center_x - (max_dim + area) // 2
        y = center_y - (max_dim + area) // 2
        roi_cropped = roi[y:y+max_dim+area, x:x+max_dim+area]   # Crops the region by taking the top left corner and slicing only the pixels within the width of our extra area
  
        # Resize the ROI to a fixed size
        fixed_size = (32, 32)
        roi_resized = cv2.resize(roi_cropped, fixed_size)
        
        # Display the resized ROI (new code)
        #  roi_resized_large = cv2.resize(roi_resized, (320, 320))  # Make it larger for better visibility
        #  cv2.imshow('Resized Square', roi_resized_large)
        #  cv2.waitKey(1)  # Small delay to show window


        # Extract the R, G, and B pixel values from the resized ROI
        r_values = roi_resized[:, :, 2].flatten()
        g_values = roi_resized[:, :, 1].flatten()
        b_values = roi_resized[:, :, 0].flatten()
        
        # Combine the R, G, and B values into a single vector
        rgb_vector = np.concatenate((r_values, g_values, b_values))
        
        # Edge detection
        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200).flatten()
        
        # Combine RGB and edge features
        features = np.concatenate((rgb_vector, edges))
        
        # Return the combined feature vector
        return features

    def train(self, model_type='svm'):
        X = np.array(self.training_data)
        y = np.array(self.labels)
        X, y = shuffle(X, y, random_state=42)
        X_scaled = self.scaler.fit_transform(X)
        if model_type == 'svm':
            self.model = SVC(kernel='linear', probability=True)
        elif model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000)
        elif model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            rf = RandomForestClassifier()
            grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
            grid_search.fit(X_scaled, y)
            self.model = grid_search.best_estimator_
            print("Best parameters found: ", grid_search.best_params_)
        elif model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            gb = GradientBoostingClassifier()
            grid_search = GridSearchCV(estimator=gb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
            grid_search.fit(X_scaled, y)
            self.model = grid_search.best_estimator_
            print("Best parameters found: ", grid_search.best_params_)
        self.model.fit(X_scaled, y)
        self.is_trained = True

    def predict(self, frame):
        if not self.is_trained:
            raise RuntimeError("Model needs to be trained first")
        h_lines, v_lines, corners = self.get_square_points()
        predictions = np.zeros((8, 8), dtype=int)
        for i in range(8):
            for j in range(8):
                square_corners = corners[i * 8 + j]
                stats = self.analyze_square(frame, square_corners)
                stats_scaled = self.scaler.transform([stats])
                predictions[i][j] = self.model.predict(stats_scaled)[0]
        return predictions

    def get_square_points(self):
        if self.h_lines is not None and self.v_lines is not None:
            return self.h_lines, self.v_lines, self.corners
            
        if len(self.points) != 4:
            raise RuntimeError("Need 4 corner points for calibration")

        pts = np.float32(self.points)
        board_width = max(
            np.linalg.norm(pts[1] - pts[0]),  # top edge
            np.linalg.norm(pts[2] - pts[3])   # bottom edge
        )
        board_height = max(
            np.linalg.norm(pts[3] - pts[0]),  # left edge
            np.linalg.norm(pts[2] - pts[1])   # right edge
        )

        # Define source and destination points for perspective transform
        src_points = pts
        dst_points = np.float32([
            [0, 0],
            [board_width, 0],
            [board_width, board_height],
            [0, board_height]
        ])

        # Get perspective transform matrix
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        inv_perspective_matrix = cv2.getPerspectiveTransform(dst_points, src_points)

        h_lines = []
        v_lines = []
        corners = []

        # Generate horizontal and vertical lines
        for i in range(9):  # 9 lines (8 squares + 1)
            h_line = []
            v_line = []
            y = i * board_height / 8
            for j in range(9):
                x = j * board_width / 8
                # Get point for horizontal line
                h_point = cv2.perspectiveTransform(
                    np.array([[[x, y]]], dtype=np.float32),
                    inv_perspective_matrix
                )[0][0]
                h_line.append(h_point)
                
                # Get point for vertical line
                v_point = cv2.perspectiveTransform(
                    np.array([[[x, y]]], dtype=np.float32),
                    inv_perspective_matrix
                )[0][0]
                v_line.append(v_point)
            h_lines.append(h_line)
            v_lines.append(v_line)

        # Generate corners for each square
        for i in range(8):
            for j in range(8):
                square_corners = [
                    h_lines[i][j],      # Top-left
                    h_lines[i][j+1],    # Top-right
                    h_lines[i+1][j+1],  # Bottom-right
                    h_lines[i+1][j]     # Bottom-left
                ]
                corners.append(square_corners)

        self.h_lines = h_lines
        self.v_lines = v_lines
        self.corners = corners
        
        return h_lines, v_lines, corners

    def start_calibration(self):
        self.points = []
        self.is_calibrating = True
        self.calibration_complete = False

    def add_calibration_point(self, x, y):
        if len(self.points) < 4:
            self.points.append((x, y))
            
        if len(self.points) == 4:
            self.calibration_complete = True
            self.is_calibrating = False
            return True
        return False

    def draw_calibration(self, frame):
        # Draw points
        for i, point in enumerate(self.points):
            cv2.circle(frame, point, 5, (0, 0, 255), -1)
            cv2.putText(frame, str(i+1), (point[0]+10, point[1]+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        # Draw lines between points
        if len(self.points) > 1:
            # Draw lines in order: 1->2->3->4->1
            for i in range(len(self.points)):
                if i + 1 < len(self.points):
                    cv2.line(frame, self.points[i], self.points[i+1], (0, 255, 0), 2)
                elif len(self.points) == 4:  # Close the quadrilateral
                    cv2.line(frame, self.points[3], self.points[0], (0, 255, 0), 2)
        
        if len(self.points) < 4:
            msg = f"Click corner point {len(self.points)+1}"
            cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2)

    def process_frame(self, frame):
        if len(self.points) != 4:
            return frame

        h_lines, v_lines, corners = self.get_square_points()
        
        # Create a copy for drawing corners on
        display_frame = frame.copy()
        
        # Draw corner points
        for square_corners in corners:
            for pt in square_corners:
                cv2.circle(display_frame, tuple(map(int, pt)), 3, (0, 0, 255), -1)

        return display_frame

    def save_data(self, filename):
        data = {
            'points': self.points,
            'training_data': self.training_data,
            'labels': self.labels,
            'scaler': self.scaler,
            'model': self.model
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {filename}")

    def mouse_clicky(self, event, x, y, flags, param):
        detector = param
        if event == cv2.EVENT_LBUTTONDOWN:
            if detector.is_calibrating:
                if detector.add_calibration_point(x, y):
                    print("Calibration complete! Now collecting training data...")
                    print("Press 'T' when ready to train the detector")
            else:
                print(f"Clicked at ({x, y})")
                for i, points in enumerate(self.corners):
                    for j, p in enumerate(points):
                        if np.linalg.norm(np.array(p) - np.array([x, y])) < 10:
                            self.selected_point = (i, j)
                            print(f"Selected point: {self.selected_point}")
                            break
        elif event == cv2.EVENT_MOUSEMOVE and self.selected_point is not None:
            i, j = self.selected_point
            self.update_shared_corners(i, j, (x, y))
            print(f"Moved point {self.selected_point} to ({x, y})")
        elif event == cv2.EVENT_LBUTTONUP:
            if self.selected_point is not None:
                i, j = self.selected_point
                self.update_shared_corners(i, j, (x, y))
                self.selected_point = None
                # print(f"Updated corners: {self.corners}")
                # for idx, corner in enumerate(self.corners):
                #    print(f"Square {idx}: {corner}")


    def update_shared_corners(self, i, j, new_point):
        # Update the selected corner
        self.corners[i][j] = new_point
        print(f"Updated corner ({i}, {j}) to {new_point}")

        # Determine the row and column of the selected square
        row = i // 8
        col = i % 8

        # Update the shared corners
        if j == 0:  # Top-left corner
            if col > 0:
                self.corners[i - 1][1] = new_point  # Top-right of the square to the left
                print(f"Updated corner ({i - 1}, 1) to {new_point}")
            if row > 0:
                self.corners[i - 8][3] = new_point  # Bottom-left of the square above
                print(f"Updated corner ({i - 8}, 3) to {new_point}")
            if col > 0 and row > 0:
                self.corners[i - 9][2] = new_point  # Bottom-right of the square diagonally above-left
                print(f"Updated corner ({i - 9}, 2) to {new_point}")
        elif j == 1:  # Top-right corner
            if col < 7:
                self.corners[i + 1][0] = new_point  # Top-left of the square to the right
                print(f"Updated corner ({i + 1}, 0) to {new_point}")
            if row > 0:
                self.corners[i - 8][2] = new_point  # Bottom-right of the square above
                print(f"Updated corner ({i - 7}, 2) to {new_point}")
            if col < 7 and row > 0:
                self.corners[i - 7 + 1][3] = new_point  # Bottom-left of the square diagonally above-right
                print(f"Updated corner ({i - 6}, 3) to {new_point}")
        elif j == 2:  # Bottom-right corner
            if col < 7:
                self.corners[i + 1][3] = new_point  # Bottom-left of the square to the right
                print(f"Updated corner ({i + 1}, 3) to {new_point}")
            if row < 7:
                self.corners[i + 8][1] = new_point  # Top-right of the square below
                print(f"Updated corner ({i + 8}, 1) to {new_point}")
            if col < 7 and row < 7:
                self.corners[i + 9][0] = new_point  # Top-left of the square diagonally below-right
                print(f"Updated corner ({i + 9}, 0) to {new_point}")
        elif j == 3:  # Bottom-left corner
            if col > 0:
                self.corners[i - 1][2] = new_point  # Bottom-right of the square to the left
                print(f"Updated corner ({i - 1}, 2) to {new_point}")
            if row < 7:
                self.corners[i + 8][0] = new_point  # Top-left of the square below
                print(f"Updated corner ({i + 8}, 0) to {new_point}")
            if col > 0 and row < 7:
                self.corners[i + 7][1] = new_point  # Top-right of the square diagonally below-left
                print(f"Updated corner ({i + 7}, 1) to {new_point}")
            

    def load_data(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.points = data['points']
        self.training_data = data['training_data']
        self.labels = data['labels']
        self.scaler = data['scaler']
        self.model = data['model']
        self.is_trained = True
        self.calibration_complete = True
        self.is_calibrating = False
        print(f"Data loaded from {filename}")

    


def main():
    cap = cv2.VideoCapture(0)
    
    # Set the resolution to 1080p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    detector = ChessPieceDetector()
    frame_count = 0

    print("Chess Piece Detector started.")
    print("Please calibrate the board first:")
    print("1. Click the top-left corner")
    print("2. Click the top-right corner")
    print("3. Click the bottom-right corner")
    print("4. Click the bottom-left corner")
    print("Press '1' to capture frame with no pieces.")
    print("Press '2' to capture frame with pieces on lower half.")
    print("Press '3' to capture frame with pieces on upper half.")
    print("Press '4' to capture frame with normal chess start.")
    print("Press 't' to train the model.")
    print("Press 's' to save the data.")
    print("Press 'l' to load the data.")
    print("Press 'q' to quit.")

    cv2.namedWindow('Chess Piece Detector')
    cv2.setMouseCallback('Chess Piece Detector', detector.mouse_clicky, detector)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam. Exiting...")
            break

        # Create display frame for visualization
        display_frame = frame.copy()

        if detector.is_calibrating:
            detector.draw_calibration(display_frame)
        else:
            display_frame = detector.process_frame(display_frame)
            if not detector.is_trained:
                cv2.putText(display_frame, "Model not trained", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                predictions = detector.predict(frame)  # Use clean frame for predictions
                # Draw predictions on display frame
                for i in range(8):
                    for j in range(8):
                        if predictions[i][j] == 1:
                            corners = [
                                detector.h_lines[i][j],
                                detector.h_lines[i][j+1],
                                detector.h_lines[i+1][j+1],
                                detector.h_lines[i+1][j]
                            ]
                            center_x = int(sum([pt[0] for pt in corners]) / 4)
                            center_y = int(sum([pt[1] for pt in corners]) / 4)
                            cv2.putText(display_frame, "X", (center_x, center_y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Chess Piece Detector', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            print("Capturing frame with no pieces...")
            empty_board = np.zeros((8, 8), dtype=int)
            detector.collect_training_data(frame, empty_board)
        elif key == ord('2'):
            print("Capturing frame with pieces on lower half...")
            lower_half_board = np.zeros((8, 8), dtype=int)
            lower_half_board[4:8, :] = 1
            detector.collect_training_data(frame, lower_half_board)
        elif key == ord('3'):
            print("Capturing frame with pieces on upper half...")
            upper_half_board = np.zeros((8, 8), dtype=int)
            upper_half_board[0:4, :] = 1
            detector.collect_training_data(frame, upper_half_board)
        elif key == ord('4'):
            print("Capturing frame with normal chess start...")
            normal_start_board = np.zeros((8, 8), dtype=int)
            normal_start_board[0:2, :] = 1
            normal_start_board[6:8, :] = 1
            detector.collect_training_data(frame, normal_start_board)
        elif key == ord('t'):
            print("Training the model...")
            detector.train()
            print("Model trained successfully!")
        elif key == ord('s'):
            filename = input("Enter filename to save data: ")
            detector.save_data(filename)
        elif key == ord('l'):
            filename = input("Enter filename to load data: ")
            detector.load_data(filename)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
