#  CHESS PIECE DETECTION AND ROBOT ARM CONTROL
#  This  allows you to play chess agaisnt a robot arm.
# Execution commands:
# start by making sure webcam has permisions: ls -l /dev/video*
# give your device sudo chmod 666 /dev/video2    -- adn 3 if you have 3 webcams 0 and 1 if its the only camera
# catkin_make
# source devel/setup.bash
# make sure your microcontroller has the correct permissions: sudo chmod 666 /dev/ttyUSB0
#roslaunch open_manipulator_controller open_manipulator_controller.launch use_platform:=true
# roslaunch chessML chessPieceDetector.launch
 
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
import chess
import pyautogui
import pyperclip
import os
import re
import rospy
from std_msgs.msg import String
from std_msgs.msg import Float64
import json  
from scipy.spatial.transform import Rotation, Slerp

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
        self.game_tracker = chess.Board()
        self.is_whites_turn = True
        self.black_move_counter = 1
        self.last_frame = None
        self.movement_threshold = 3510062 
        
        # New: set the folder in which training data samples will be saved
        self.training_folder = "/home/l/catkin_ws/src/chessML/src/training_samples"
        if not os.path.exists(self.training_folder):
            os.makedirs(self.training_folder)
        
        self.arena_controller = ChessArenaController()
        self.arena_calibration_mode = False



    def collect_training_data(self, frame, label):
        h_lines, v_lines, corners = self.get_square_points()
        
        # Add rotation variations
        angles = [0, 5, -5, 10, -10]
        for angle in angles:
            rotated_frame = self._rotate_frame(frame, angle) if angle != 0 else frame
            rotated_corners = self._rotate_corners(corners, angle, frame.shape) if angle != 0 else corners
            for i in range(8):
                for j in range(8):
                    square_corners = rotated_corners[i * 8 + j]
                    stats = self.analyze_square(rotated_frame, square_corners)
                    self.training_data.append(stats)
                    self.labels.append(label[i][j])
                    
                    # Save the training sample to disk.
                    sample = {'features': stats, 'label': label[i][j]}
                    sample_filename = os.path.join(self.training_folder, f"sample_{int(time.time()*1000)}_{i}_{j}_{angle}.pkl")
                    with open(sample_filename, 'wb') as f:
                        pickle.dump(sample, f)

    def _rotate_frame(self, frame, angle):
        
        #Rotate the frame by the specified angle around its center.
        
        height, width = frame.shape[:2]
        center = (width/2, height/2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(frame, rotation_matrix, (width, height))

    def _rotate_corners(self, corners, angle, shape):
        # Rotate corner points to match the rotated frame
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

    def train(self, model_type='logistic'):
        # Load all training samples from the training folder
        training_vectors = []
        training_labels = []
        for filename in os.listdir(self.training_folder):
            filepath = os.path.join(self.training_folder, filename)
            with open(filepath, 'rb') as f:
                sample = pickle.load(f)
                training_vectors.append(sample['features'])
                training_labels.append(sample['label'])
        
        if not training_vectors:
            raise RuntimeError("No training data found in the folder.")
        
        X = np.array(training_vectors)
        y = np.array(training_labels)
        X, y = shuffle(X, y, random_state=42)
        X_scaled = self.scaler.fit_transform(X)
        
        if model_type == 'svm':
            self.model = SVC(kernel='linear', probability=True)
        elif model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
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
        
        if self.arena_calibration_mode:
            msg = "Arena Calibration Mode - "
            if len(self.arena_controller.arena_corners) < 4:
                point_names = ["top-left", "top-right", "bottom-right", "bottom-left"]
                msg += f"Click {point_names[len(self.arena_controller.arena_corners)]} corner"
            cv2.putText(frame, msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 255), 2)
            
            # Draw already placed points
            for i, point in enumerate(self.arena_controller.arena_corners):
                cv2.circle(frame, point, 5, (0, 255, 255), -1)
                cv2.putText(frame, str(i+1), (point[0]+10, point[1]+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

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

    def board_to_array(self):
        #Convert current chess.Board state to numpy array (0=empty, 1=piece)
        board_array = np.zeros((8, 8), dtype=int)
        for i in range(8):
            for j in range(8):
                square = chess.square(j, 7-i)
                piece = self.game_tracker.piece_at(square)
                if piece is not None:
                    board_array[i][j] = 1
        return board_array

    def detect_move(self, current_state):
        #Detect move by comparing current state with internal game state
        board_state = self.board_to_array()
        diff = current_state != board_state
        changed_squares = list(zip(*np.where(diff)))
        
        print("\nBoard states:")
        print("Current detected state:")
        print(current_state)
        print("Internal game state:")
        print(board_state)
        print("Current chess board position:")
        print(self.game_tracker)
        
        # Find squares where pieces disappeared and appeared
        from_squares = [(r, c) for r, c in changed_squares 
                       if current_state[r][c] == 0 and board_state[r][c] == 1]
        to_squares = [(r, c) for r, c in changed_squares 
                     if current_state[r][c] == 1 and board_state[r][c] == 0]
        
        print(f"Pieces moved from: {from_squares}")
        print(f"Pieces moved to: {to_squares}")
        # Check for castling patterns
        # White kingside castling - e1 and h1 empty, g1 and f1 occupied
        if (7, 4) in from_squares and (7, 7) in from_squares and (7, 5) in to_squares and (7, 6) in to_squares:
            print("\nWhite kingside castling detected!")
            # Return king's from square and rook's from square
            return ((7, 4), (7, 6))
        
        # White queenside castling - e1 and a1 empty, c1 and d1 occupied
        if (7, 4) in from_squares and (7, 0) in from_squares and (7, 2) in to_squares and (7, 3) in to_squares:
            print("\nWhite queenside castling detected!")
            # Return king's from square and rook's from square
            return ((7, 4), (7, 1))
        # Try regular moves FIRST
        for from_square in from_squares:
            for to_square in to_squares:
                move_uci = self.square_to_algebraic(*from_square) + self.square_to_algebraic(*to_square)
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in self.game_tracker.legal_moves and not self.game_tracker.is_capture(move):
                        print(f"\nNormal move detected: {move_uci}")
                        return (from_square, to_square)
                except ValueError:
                    continue
        
        # Check captures if no normal moves found
        for from_square in from_squares:
            from_algebraic = self.square_to_algebraic(*from_square)
            from_chess_square = chess.parse_square(from_algebraic)
            
            piece = self.game_tracker.piece_at(from_chess_square)
            if piece:
                print(f"\nChecking captures for {piece} at {from_algebraic}")
                
                # Track best capture move
                best_move = None
                best_target_value = -float('inf')
                lowest_diff = float('inf')
                
                # Check all legal captures from this square
                for move in self.game_tracker.legal_moves:
                    if move.from_square == from_chess_square and self.game_tracker.is_capture(move):
                        to_algebraic = chess.square_name(move.to_square)
                        print(f"Found capture possibility: {from_algebraic} -> {to_algebraic}")
                        
                        # Evaluate target piece including positional value
                        target_piece = self.game_tracker.piece_at(move.to_square)
                        target_value = self.get_piece_value(target_piece, move.to_square)
                        print(f"Target value at {to_algebraic}: {target_value}")
                        
                        # Test this capture
                        test_board = self.game_tracker.copy()
                        test_board.push(move)
                        test_state = self.board_to_array_from_board(test_board)
                        
                        # Compare board states after capture
                        differences = np.sum(np.abs(test_state - current_state))
                        print(f"Differences after capture {move}: {differences}")
                        
                        # Update best move if this is better
                        if differences <= 2:  # Allow for some detection noise
                            if best_move is None or target_value > best_target_value:
                                best_move = move
                                best_target_value = target_value
                                lowest_diff = differences
                
                # If we found a valid capture
                if best_move:
                    to_square = (
                        8 - int(chess.square_name(best_move.to_square)[1]),
                        ord(chess.square_name(best_move.to_square)[0]) - ord('a')
                    )
                    print(f"\nBest capture move found: {best_move} (value: {best_target_value}, differences: {lowest_diff})")
                    return (from_square, to_square)
        
        return None

    def get_piece_value(self, piece, square):
        """Get the material and positional value of a piece"""
        if piece is None:
            return 0
            
        # Base material values
        base_values = {
            'P': 1.0,  # Pawn
            'N': 3.0,  # Knight
            'B': 3.0,  # Bishop
            'R': 5.0,  # Rook
            'Q': 9.0,  # Queen
            'K': 0.0   # King
        }
        
        # Positional bonus tables
        pawn_position = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.1, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.1],
            [0.05, 0.05, 0.1, 0.25, 0.25, 0.1, 0.05, 0.05],
            [0.0, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0],
            [0.05, -0.05, -0.1, 0.0, 0.0, -0.1, -0.05, 0.05],
            [0.05, 0.1, 0.1, -0.2, -0.2, 0.1, 0.1, 0.05],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ]
        
        piece_symbol = piece.symbol().upper()
        base_value = base_values.get(piece_symbol, 0)
        
        # Add positional bonus based on piece type and square
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        
        positional_bonus = 0.0
        if piece_symbol == 'P':
            positional_bonus = pawn_position[rank][file]
            # Extra bonus for central pawns
            if file in [3, 4] and rank in [3, 4]:
                positional_bonus += 0.3
        elif piece_symbol in ['N', 'B']:
            # Bonus for central control
            if file in [2, 3, 4, 5] and rank in [2, 3, 4, 5]:
                positional_bonus += 0.2
        
        return base_value + positional_bonus

    def board_to_array_from_board(self, board):
        #Convert a chess.Board to numpy array (0=empty, 1=piece
        board_array = np.zeros((8, 8), dtype=int)
        for i in range(8):
            for j in range(8):
                square = chess.square(j, 7-i)
                piece = board.piece_at(square)
                if piece is not None:
                    board_array[i][j] = 1
        return board_array



    def square_to_algebraic(self, row, col):
        #Convert row,col coordinates to algebraic notation
        file = chr(ord('a') + col)
        rank = str(8 - row)
        return file + rank

    def execute_move(self, move_squares):
        """Execute detected move using Arena GUI controls"""
        from_square = move_squares[0]
        to_square = move_squares[1]
        
        if not self.arena_controller.is_calibrated:
            print("Arena board not calibrated! Cannot execute move.")
            return False

        move = chess.Move.from_uci(
            self.square_to_algebraic(*from_square) + 
            self.square_to_algebraic(*to_square)
        )
        
        if move in self.game_tracker.legal_moves:
            try:
                # Find and activate Arena window

                
    
                print("Activating Arena window...")
                time.sleep(0.5)
                
                # Execute move using mouse
                print(f"Moving piece from {from_square} to {to_square}")
                self.arena_controller.execute_move(from_square, to_square)
                
                # Update internal game state
                self.game_tracker.push(move)
                self.is_whites_turn = False
                return True
  
                    
            except Exception as e:
                print(f"Error executing move: {e}")
                return False
        
        return False

    # def capture_black_move(self):
    #     #Capture black's move from clipboard and update game state
    #     # Wait for move to be copied
    #     pyautogui.press('f5')  # Refresh move list
    #     time.sleep(1)  # Wait for clipboard update
    #     from_square = ""
    #     to_square = ""
    #     # Get move from clipboard
    #     clipboard = pyperclip.paste()
    #     move_str = self.extract_black_move(clipboard)
        
    #     print("\nDEBUG Black's Move:")
    #     print("Clipboard content:", clipboard)
    #     print("Extracted move:", move_str)
        
    #     if move_str:
    #         uci_move = self.convert_to_uci(move_str)
          
    
    #         try:
    #             move = chess.Move.from_uci(uci_move)
    #             piece = self.game_tracker.piece_at(move.from_square)

    #             from_square = (7 - chess.square_rank(move.from_square), chess.square_file(move.from_square))
    #             to_square =  (7 - chess.square_rank(move.to_square), chess.square_file(move.to_square))
    #             print("from and to square from chess library: ", from_square, " " , to_square, " piece=", piece)
    #             if move in self.game_tracker.legal_moves:
    #                 self.game_tracker.push(move)
    #                 print("\nUpdated board after black's move:")
    #                 print(self.game_tracker)
    #                 print("\nBoard state array:")
    #                 print(self.board_to_array())
    #                 self.is_whites_turn = True
    #         except ValueError as e:
    #             print(f"Error processing black's move: {e}")
        
    #     return False

    def capture_black_move(self):
        """
        Capture black's move from clipboard, update game state, and publish the move.
        """
        # Refresh move list and wait for clipboard update
        pyautogui.press('f5')  # Refresh move list
        time.sleep(1)  # Wait for clipboard update

        # Get move from clipboard
        clipboard = pyperclip.paste()
        move_str = self.extract_black_move(clipboard)

        print("\nDEBUG Black's Move:")
        print("Clipboard content:", clipboard)
        print("Extracted move:", move_str)

        if move_str:
            uci_move = self.convert_to_uci(move_str)

            try:
                move = chess.Move.from_uci(uci_move)
                piece = self.game_tracker.piece_at(move.from_square)

                # Convert from and to squares to array locations
                from_square = (7 - chess.square_rank(move.from_square), chess.square_file(move.from_square))
                to_square = (7 - chess.square_rank(move.to_square), chess.square_file(move.to_square))
                print("From and to square from chess library: ", from_square, " ", to_square, " piece=", piece)

                if move in self.game_tracker.legal_moves:
                    # Check if the move is a capture BEFORE pushing the move
                    is_capture = self.game_tracker.is_capture(move)

                    # Update the internal game state
                    self.game_tracker.push(move)
                    print("\nUpdated board after black's move:")
                    print(self.game_tracker)
                    print("\nBoard state array:")
                    print(self.board_to_array())
                    self.is_whites_turn = True

                    # Convert numerical positions to chess notation (e.g., (1, 3) -> 'd2')
                    start_notation = chr(from_square[1] + ord('a')) + str(8 - from_square[0])
                    end_notation = chr(to_square[1] + ord('a')) + str(8 - to_square[0])

                    # Combine into UCI format (e.g., 'd2d4')
                    uci_move = start_notation + end_notation

                    # Append 'x' to the move if it's a capture
                    if is_capture:
                        uci_move += 'x'


                    return from_square, to_square, piece, uci_move

            except ValueError as e:
                print(f"Error processing black's move: {e}")

        return False




    def extract_black_move(self, clipboard_content):
        # Extract black's move from clipboard text"""
        print("\nDEBUG Move Extraction:")
        print("Full clipboard content:")
        print(clipboard_content)
        
        
        # Find all moves first
        moves_section = ""
        in_moves = False
        
        for line in clipboard_content.split('\n'):
            # Skip empty lines
            if not line.strip():
                continue
            # Start of moves section (first line with move number)
            if re.match(r'^\d+\.', line.strip()):
                in_moves = True
            # If we're in moves section, add the line
            if in_moves:
                moves_section += line.strip() + " "
        
        if not moves_section:
            print("No moves found in content")
            return None
            
        print(f"Moves section: {moves_section}")
        
        # Clean up annotations and comments
        clean_moves = re.sub(r'\{[^}]*\}', '', moves_section)  # Remove comments
        clean_moves = clean_moves.split('*')[0]  # Remove end of game marker
        tokens = clean_moves.strip().split()
        
        # Find the last complete move
        last_black_move = None
        current_move_number = 0
        
        for i, token in enumerate(tokens):
            if token.endswith('.'):  # This is a move number
                try:
                    move_num = int(token[:-1])
                    if move_num > current_move_number:
                        current_move_number = move_num
                        # Check if there's both white and black moves
                        if i + 2 < len(tokens):
                            white_move = tokens[i + 1]
                            black_move = tokens[i + 2]
                            last_black_move = black_move
                except ValueError:
                    continue
        
        if last_black_move:
            print(f"Found last black move: {last_black_move}")
            return last_black_move
        
        print("No complete black move found")
        return None

    
    def convert_to_uci(self, move_text):
        """Convert algebraic notation to UCI format"""
        print(f"Converting move: {move_text}")
        
        # Handle special characters and cleanup
        move_text = move_text.replace('*', '').strip()
        
        try:
            # Let python-chess do the conversion
            move = self.game_tracker.parse_san(move_text)
            uci_move = move.uci()
            print(f"Converted {move_text} to UCI: {uci_move}")
            return uci_move
        except ValueError as e:
            print(f"Failed to convert move: {move_text}")
            print(f"Error: {e}")
            return None
    
     # Detect movement in the frame
    def detect_movement(self, frame):

        if self.last_frame is None:
            self.last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame_diff = cv2.absdiff(gray, self.last_frame)
        
        self.last_frame = gray

        movement = np.sum(frame_diff) > self.movement_threshold
        print(np.sum(frame_diff))
        cv2.putText(frame, str(movement), (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return movement

class ChessArenaController:
    def __init__(self):
        self.arena_corners = []
        self.ideal_corners = np.float32([[0, 0], [640, 0], [640, 640], [0, 640]])
        self.perspective_matrix = None
        self.is_calibrated = False
        self.calibrating = False
        self.board_size = None
        self.square_size = None
        self.board_width = None
        self.board_height = None

    def calculate_board_size(self):
        """Calculate actual board size from calibration points"""
        if len(self.arena_corners) != 4:
            return

        corners = np.array(self.arena_corners)
        
        # Calculate width and height
        self.board_width = max(
            np.linalg.norm(corners[1] - corners[0]),  # top edge
            np.linalg.norm(corners[2] - corners[3])   # bottom edge
        )
        self.board_height = max(
            np.linalg.norm(corners[3] - corners[0]),  # left edge
            np.linalg.norm(corners[2] - corners[1])   # right edge
        )
        
        self.board_size = min(self.board_width, self.board_height)
        self.square_size = self.board_size / 8
        
        print(f"Calculated board dimensions: {self.board_width:.1f}px x {self.board_height:.1f}px")
        print(f"Square size: {self.square_size:.1f}px")

    def get_square_center(self, square):
        """Get screen coordinates for the center of a chess square"""
        if not self.board_size or not self.square_size:
            raise RuntimeError("Board size not calculated. Complete calibration first.")
            
        row, col = square
        
        # Normalize coordinates to [0,1] range
        x = (col + 0.5) / 8.0  # Add 0.5 to get center of square
        y = (row + 0.5) / 8.0
        
        # Convert to actual screen coordinates using linear interpolation
        screen_x = int(self.arena_corners[0][0] + x * (self.arena_corners[1][0] - self.arena_corners[0][0]))
        next_y = self.arena_corners[3][1] - self.arena_corners[0][1]
        screen_y = int(self.arena_corners[0][1] + y * next_y)
        
        print(f"Square {square} -> Normalized pos ({x:.2f}, {y:.2f}) -> Screen pos ({screen_x}, {screen_y})")
        return (screen_x, screen_y)

    def execute_move(self, from_square, to_square):
        """Execute move on Arena board by clicking the squares"""
        try:
            # Get screen coordinates for the squares
            from_pos = self.get_square_center(from_square)
            to_pos = self.get_square_center(to_square)
            
            print(f"Moving piece from {from_square} -> {to_square}")
            print(f"Screen coordinates: {from_pos} -> {to_pos}")
            
            # Disable PyAutoGUI failsafe temporarily
            original_failsafe = pyautogui.FAILSAFE
            pyautogui.FAILSAFE = False
            
            try:
                # Click source square
                pyautogui.moveTo(from_pos[0], from_pos[1], duration=0.2)
                time.sleep(0.1)
                pyautogui.mouseDown()
                time.sleep(0.1)
                
                # Move to destination square
                pyautogui.moveTo(to_pos[0], to_pos[1], duration=0.3)
                time.sleep(0.1)
                
                # Release to complete the move
                pyautogui.mouseUp()
                time.sleep(0.1)
            finally:
                # Restore original failsafe setting
                pyautogui.FAILSAFE = original_failsafe
            
            return True
            
        except Exception as e:
            print(f"Error executing move: {e}")
            return False

    def add_corner(self):
        """Add current mouse position as a corner point"""
        x, y = pyautogui.position()
        point_names = ["top-left", "top-right", "bottom-right", "bottom-left"]
        current_point = len(self.arena_corners)
        
        self.arena_corners.append((x, y))
        print(f"Arena calibration: Added {point_names[current_point]} point at ({x, y})")
        
        if len(self.arena_corners) == 4:
            # Calculate board size first
            self.calculate_board_size()
            
            # Then setup perspective transform
            pts = np.float32(self.arena_corners)
            self.perspective_matrix = cv2.getPerspectiveTransform(self.ideal_corners, pts)
            self.is_calibrated = True
            self.calibrating = False
            
            print("\n=== Arena Board Calibration Complete! ===")
            print("Corner points:")
            for i, (px, py) in enumerate(self.arena_corners):
                print(f"  {point_names[i]}: ({px}, {py})")
            return True
        else:
            remaining = 4 - len(self.arena_corners)
            if remaining > 0:
                print(f"Need {remaining} more points. Next point: {point_names[len(self.arena_corners)]}")
        return False

    def start_calibration(self):
        """Start the calibration process"""
        self.arena_corners = []
        self.calibrating = True
        self.is_calibrated = False
        self.board_size = None
        self.square_size = None
        print("\n=== Starting Arena Board Calibration ===")
        print("Move mouse to each corner of the Arena board and press SPACE:")
        print("1. Top-left corner")
        print("2. Top-right corner")
        print("3. Bottom-right corner")
        print("4. Bottom-left corner")
        print("Press ESC to cancel calibration")

class RobotChessController:
    def __init__(self):
        # check if IKFK is available
        self.is_initialized = False
        
        # Initialize ROS publishers for joint control 
        self.joint1_pub = rospy.Publisher('/joint1_position/command', Float64, queue_size=10)
        self.joint2_pub = rospy.Publisher('/joint2_position/command', Float64, queue_size=10)
        self.joint3_pub = rospy.Publisher('/joint3_position/command', Float64, queue_size=10)
        self.joint4_pub = rospy.Publisher('/joint4_position/command', Float64, queue_size=10)
        self.gripper_pub = rospy.Publisher('/gripper_position/command', Float64, queue_size=10)
        
        # Hard-coded joint angles for calibrated positions
        # Format: [base, joint1, joint2, joint3, joint4, end_effector]
        self.calibrated_positions = {
            # Top-left corner (a8)
            'a8': [0, -1.230, -0.236, 0.571,  1.278, 0],  
            # Top-right corner (h8)
            'h8': [0, 1.269, -0.199, 0.523, 1.261, 0],    
            # Bottom-right corner (h1)
            'h1': [0, 0.422, 0.798, -0.977, 1.470, 0],   
            # Bottom-left corner (a1)
            'a1': [0, -0.393, 0.816, -0.999, 1.470, 0],

            # Center squares
            'd4': [0, -0.077, 0.089, 0.176, 1.325, 0],     
            'd5': [0, -0.098, -0.057, 0.376, 1.325, 0],     
            'e4': [0, 0.115, 0.103, 0.161, 1.328, 0],      
            'e5': [0, 0.129, -0.043, 0.371, 1.331, 0],      
            
            # Middle edges
            'd8': [0, -0.282, -0.962, 0.983, 1.559, 0],    
            'd1': [0, -0.051, 0.683, -0.761, 1.482, 0],     
            'a4': [0, -0.584, 0.451, -0.336, 1.571, 0],    
            'h4': [0, 0.621, 0.465, -0.385, 1.571, 0],  

            # More random ones
            'c6': [0, -0.413, -0.279, 0.609, 1.252, 0],  
            'f6': [0, -0.479, -0.244, 0.561, 1.307, 0], 


        }
        
        
        # Store camera positions for all squares
        self.square_centers = {}
        
        # World positions corresponding to corners (calculated using FK)
        self.world_positions = {}
        self.calibrated_orientations = {}
        
        # Dictionary to store calculated positions for all squares
        self.chess_square_positions = {}
        
        # Parameters for movement
        self.hover_offset = 0.07  # 5cm above the board surface
        
        # Captured piece storage configuration
        self.captured_piece_joints = [0, 1.204, 0.476, -0.340, 1.127, 0]  # Joint angles for captured piece storage
        self.captured_piece_orientation = np.array([-0.334, 0.487, 0.457, 0.665])  # Quaternion orientation
        self.captured_piece_joints2 = [0,1.2,0.225,-0.490,1.127,0]
        self.captured_piece_orientation2 = np.array([-0.236, 0.345, 0.513, 0.75])
        
        # Try to import IKFK module but fail gracefully
        try:
            from IKFK import create_open_manipulator_chain, forward_kinematics_test, inverse_kinematics_test
            self.chain = create_open_manipulator_chain()
            self.forward_kinematics = forward_kinematics_test
            self.inverse_kinematics = inverse_kinematics_test
            self.is_initialized = True
            print("Robot controller initialized successfully")
            
            # Calculate world positions for the four corners immediately after initialization
            if self.calculate_world_positions():
                print("Successfully calculated world positions for corners")
            else:
                print("Failed to calculate world positions for corners")
        except ImportError as e:
            print(f"Robot control disabled: {e}")
    
    def update_from_detector_corners(self, detector_corners):
        """
        Update square centers using the corner points from ChessPieceDetector.
        
        Args:
            detector_corners: The corners array from ChessPieceDetector, containing
                             corner coordinates for all 64 squares
        """
        if len(detector_corners) != 64:
            print(f"Error: Expected 64 squares, got {len(detector_corners)}")
            return False
        
        # Clear previous centers
        self.square_centers = {}
        
        # Files and ranks for chess notation
        files = 'abcdefgh'
        ranks = '87654321'
        
        # Calculate center point of each square
        for i in range(8):
            for j in range(8):
                square_idx = i * 8 + j
                corners = detector_corners[square_idx]
                
                # Calculate center as average of four corners
                center_x = sum(corner[0] for corner in corners) / 4
                center_y = sum(corner[1] for corner in corners) / 4
                
                # Convert to chess notation
                square = files[j] + ranks[i]
                
                # Store the center point
                self.square_centers[square] = (center_x, center_y)
               
                #print(f"Square {square} -> Center ({center_x}, {center_y})")
                
        print(f"Updated center points for all 64 squares from detector corners")
        
        # Map important corners to our world positions
        if self.is_initialized:
            self.map_camera_to_world()
        
        return True

    def calculate_world_positions(self):
        """Calculate world coordinates from calibrated joint angles"""
        if not self.is_initialized:
            return False
        
        for square, angles in self.calibrated_positions.items():
            print(f"Calculating world position for {square} with angles {angles}")
            fk_result = self.forward_kinematics(angles)
            self.world_positions[square] = fk_result["position"]
            self.calibrated_orientations[square] = fk_result["orientation"]
        
        print("Corner world positions calculated from joint angles:")
        for square, pos in self.world_positions.items():
            print(f"  {square}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
        
        return True

    def map_camera_to_world(self):
        """Map camera center points to world positions using our calibrated corners"""
        if not self.is_initialized:
            return False
            
        if len(self.square_centers) != 64:
            print("Square centers not fully populated")
            return False
            
        # Get the camera center points
        camera_centers = {
            'a8': self.square_centers.get('a8'),
            'h8': self.square_centers.get('h8'),
            'h1': self.square_centers.get('h1'),
            'a1': self.square_centers.get('a1')
        }
        
        if None in camera_centers.values():
            print("Missing camera center positions")
            return False
            
           # Convert world and camera positions to numpy arrays for easier computation
        world_corners = np.array([
            self.world_positions['a8'],  # Top-left
            self.world_positions['h8'],  # Top-right
            self.world_positions['h1'],  # Bottom-right
            self.world_positions['a1']   # Bottom-left
        ])
        
        camera_corners_array = np.array([
            camera_centers['a8'],  # Top-left
            camera_centers['h8'],  # Top-right
            camera_centers['h1'],  # Bottom-right
            camera_centers['a1']   # Bottom-left
        ])
        
        # Get corner orientations
        corner_orientations = np.array([
            self.calibrated_orientations['a8'],  # Top-left
            self.calibrated_orientations['h8'],  # Top-right
            self.calibrated_orientations['h1'],  # Bottom-right
            self.calibrated_orientations['a1']   # Bottom-left
        ])
        
        # For each square, calculate its world position
        files = 'abcdefgh'
        ranks = '87654321'
        
        for rank in ranks:
            for file in files:
                square = file + rank
                if square in self.square_centers:
                    camera_pos = self.square_centers[square]
                    x_norm, y_norm = self._get_normalized_position(camera_pos, camera_corners_array)
                    
                    # Get interpolated position and orientation
                    world_pos = self._bilinear_interpolate(world_corners, x_norm, y_norm)
                    orientation = self._interpolate_orientation(corner_orientations, x_norm, y_norm)
                    
                    # Store both position and orientation
                    self.chess_square_positions[square] = {
                        'camera': camera_pos,
                        'world': world_pos,
                        'orientation': orientation
                    }
                    print(f"Mapped {square} to {world_pos} with an orientation of {orientation}")
        
        print(f"Mapped camera centers to world positions for all 64 squares")
        return True
    
    def _get_normalized_position(self, point, corners):
        """
        Calculate the normalized position (0-1) of a point within the quadrilateral
        defined by the four corners using a perspective transform.
        
        Args:
            point: The (x, y) point to normalize
            corners: Array of four corner points [top_left, top_right, bottom_right, bottom_left]
            
        Returns:
            (x_norm, y_norm): Normalized position using perspective transform
        """
        # Define the destination points for a normalized square
        dst_points = np.array([[0,0], [1,0], [1,1], [0,1]], dtype=np.float32)
        
        # Convert corners to float32 array
        src_points = corners.astype(np.float32)
        
        # Calculate the perspective transform matrix
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Convert point to homogeneous coordinates
        point_h = np.array([[[point[0], point[1]]]], dtype=np.float32)
        
        # Apply perspective transform to get normalized coordinates
        normalized = cv2.perspectiveTransform(point_h, perspective_matrix)
        x_norm, y_norm = normalized[0][0]
        
        # Constrain to 0-1 range to handle numerical errors
        x_norm = max(0, min(1, x_norm))
        y_norm = max(0, min(1, y_norm))
        
        return x_norm, y_norm
    
    def _bilinear_interpolate(self, corners, x_norm, y_norm):
        """
        Perform bilinear interpolation between four corner points.
        
        Args:
            corners: Array of four points [top_left, top_right, bottom_right, bottom_left]
            x_norm: Normalized x position (0-1)
            y_norm: Normalized y position (0-1)
            
        Returns:
            Interpolated position
        """
        # Linear interpolation along top and bottom edges
        top = corners[0] * (1 - x_norm) + corners[1] * x_norm
        bottom = corners[3] * (1 - x_norm) + corners[2] * x_norm
        
        # Linear interpolation between top and bottom
        return top * (1 - y_norm) + bottom * y_norm
    
    def get_square_camera_position(self, square):
        """Get camera position for a chess square"""
        if square in self.square_centers:
            return self.square_centers[square]
        return None
    
    def get_square_world_position(self, square):
        """Get world position for a chess square"""
        if square in self.chess_square_positions:
            return self.chess_square_positions[square]['world']
        return None
    
    def generate_move_trajectory(self, uci_move):
        """
        Generate joint angles for moving a piece from one square to another.
        Returns a list of positions that can be sent as a message.
        """
        print(f"Generating move trajectory from {uci_move}")
        if not self.is_initialized:
            print("Robot controller not initialized")
            return None
        trajectory_points = []
        
        # Parse UCI move string (e.g. "d7d5" or "d7d5x" for captures)
        from_square = uci_move[:2]
        to_square = uci_move[2:4]  # Get destination before the 'x' if it exists
        
        if from_square not in self.chess_square_positions or to_square not in self.chess_square_positions:
            print(f"Square positions not calculated for {from_square} or {to_square}")
            return None
            
        # Get world positions for source and destination squares
        from_pos = self.chess_square_positions[from_square]['world']
        to_pos = self.chess_square_positions[to_square]['world']
        
        if uci_move.endswith('x'):
            # Handle capture - need to remove captured piece first
            print("Generating capture sequence...")
            
            # Get both pre-calibrated positions and their orientations for captured pieces
            fk_result1 = self.forward_kinematics(self.captured_piece_joints2)
            fk_result2 = self.forward_kinematics(self.captured_piece_joints)
            captured_piece_xyz1 = fk_result1["position"]
            captured_piece_xyz2 = fk_result2["position"]
            captured_piece_orientation1 = fk_result1["orientation"]
            captured_piece_orientation2 = fk_result2["orientation"]
            print(f"Captured piece storage positions: {captured_piece_xyz1}, {captured_piece_xyz2}")
            
            # Add capture sequence trajectory points
            # 1. Hover above piece to be captured
            hover_capture = np.copy(to_pos)
            hover_capture[2] += self.hover_offset
            trajectory_points.append(hover_capture)
            
            # 2. Lower to grab captured piece
            trajectory_points.append(to_pos)
            
            # 3. Raise back up
            trajectory_points.append(hover_capture)
            
            # 4. Move to first captured piece storage location
            trajectory_points.append(captured_piece_xyz1)
            
            # 5. Move to second captured piece storage location
            trajectory_points.append(captured_piece_xyz2)
            
            print("Capture sequence added to trajectory")
        
        # Add the normal move sequence
        # 1. Hover above source
        hover_from = np.copy(from_pos)
        hover_from[2] += self.hover_offset
        trajectory_points.append(hover_from)
        
        # 2. Move down to piece
        trajectory_points.append(from_pos)
        
        # 3. Move back up to hover
        trajectory_points.append(hover_from)
        
        # 4. Hover above destination
        hover_to = np.copy(to_pos)
        hover_to[2] += self.hover_offset
        trajectory_points.append(hover_to)
        
        # 5. Move down to place piece
        trajectory_points.append(to_pos)
        
        # 6. Move back up to hover
        trajectory_points.append(hover_to)
        
        # Convert each point from world coordinates to joint angles using IK
        joint_trajectory = []
        previous_angles = None  # Use as seed for next IK solution
        
        print(f"Total trajectory points: {len(trajectory_points)}")
        for i, point in enumerate(trajectory_points):
            # Get orientation based on the target square
            if uci_move.endswith('x'):
                if i == 3:  # First storage position
                    orientation = captured_piece_orientation1
                    # Use the actual joint values for this pre-calibrated position
                    joint_angles = self.captured_piece_joints2.copy()
                    joint_trajectory.append(joint_angles)
                    previous_angles = joint_angles
                    print(f"Point {i+1}: Position: {point}, Joint angles: {np.round(joint_angles[1:5], 3)} (using pre-calibrated)")
                    continue
                    
                elif i == 4:  # Second storage position
                    orientation = captured_piece_orientation2
                    # Use the actual joint values for this pre-calibrated position
                    joint_angles = self.captured_piece_joints.copy()
                    joint_trajectory.append(joint_angles)
                    previous_angles = joint_angles
                    print(f"Point {i+1}: Position: {point}, Joint angles: {np.round(joint_angles[1:5], 3)} (using pre-calibrated)")
                    continue
                    
                else:
                    # Use interpolated orientation for the current position
                    current_square = to_square if i < 3 else from_square
                    orientation = self.chess_square_positions[current_square]['orientation']
            else:
                # Use interpolated orientation for normal moves
                current_square = from_square if i < 3 else to_square
                orientation = self.chess_square_positions[current_square]['orientation']
            
            # Get IK solution with interpolated orientation
            if previous_angles is None:
                ik_result = self.inverse_kinematics(point, orientation)
            else:
                # Use previous angles as seed for next IK solution
                ik_result = self.inverse_kinematics(point, orientation, previous_angles)
                
            joint_angles = ik_result["joint_angles"]
            previous_angles = joint_angles.copy()  # Make a copy to ensure it's not modified
            
            # Add gripper control based on position
            if i in [1, 4, 7, 10]:  # Points where we need to grab/release
                joint_angles[5] = 0.0  # Close gripper
            else:
                joint_angles[5] = 0.5  # Open gripper
                
            joint_trajectory.append(joint_angles)
            
            # Print info about the point
            print(f"Point {i+1}: Position: {point}, Joint angles: {np.round(joint_angles[1:5], 3)}")
        
        print(f"Joint trajectory: {joint_trajectory}")
        
        return joint_trajectory

    def _interpolate_orientation(self, corners_orient, x_norm, y_norm):
        """
        Interpolate orientation using scipy's Slerp
        """
        # Create Rotation objects from quaternions for each corner
        rotations = Rotation.from_quat(corners_orient)
        
        # First interpolate along top and bottom edges
        top_edge = Slerp([0, 1], rotations[[0, 1]])
        bottom_edge = Slerp([0, 1], rotations[[3, 2]])
        
        # Get the rotations at x_norm position
        top_rot = top_edge(x_norm)
        bottom_rot = bottom_edge(x_norm)
        
        # Now interpolate between top and bottom
        # Create a new Rotation from the quaternions of the top and bottom rotations
        top_quat = top_rot.as_quat()
        bottom_quat = bottom_rot.as_quat()
        vertical_rots = Rotation.from_quat([top_quat, bottom_quat])
        vertical_interp = Slerp([0, 1], vertical_rots)
        
        final_rot = vertical_interp(y_norm)
        
        return final_rot.as_quat()

def json_trajectory(joint_trajectory):
    """Convert joint trajectory arrays to a JSON string format that qnode can parse"""
    # Convert numpy arrays to lists for JSON serialization
    trajectory_list = []
    for angles in joint_trajectory:
        # Convert each point to a list with clean float values (limited decimal places)
        angles_list = [float(f"{val:.6f}") for val in angles]
        trajectory_list.append(angles_list)
    return json.dumps(trajectory_list)

def main():
    # Initialize ROS node
    rospy.init_node('chess_piece_detector', anonymous=True)
    pub = rospy.Publisher('/chess_moves', String, queue_size=10)
    cap = cv2.VideoCapture(2, cv2.CAP_V4L2)

    
    # Set the resolution to 1080p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    detector = ChessPieceDetector()
    robot_controller = RobotChessController()
    frame_count = 0

    print("Chess Piece Detector started.")
    print("Please calibrate the board first:")
    print("1. Click the top-left corner")
    print("2. Click the top-right corner")
    print("3. Click the bottom-right corner")
    print("4. Click the bottom-left corner")
    print("Press '1' to capture frame with no pieces.")
    print("Press '2' to capture frame with white pieces on lower half.")
    print("Press '3' to capture frame with black pieces on upper half.")
    print("Press '4' to capture frame with normal chess start.")
    print("Press 't' to train the model.")
    print("Press 's' to save the data.")
    print("Press 'l' to load the data.")
    print("Press 'q' to quit.")
    print("Additional instructions:")
    print("Press 'a' to start Arena board calibration")
    print("When in Arena calibration mode, click the four corners of the Arena board in order:")
    print("1. Top-left")
    print("2. Top-right")
    print("3. Bottom-right")
    print("4. Bottom-left")

    def mouse_callback(event, x, y, flags, param):
        detector = param
        if event == cv2.EVENT_LBUTTONDOWN:
            if detector.arena_calibration_mode:
                if detector.arena_controller.add_corner(x, y):
                    detector.arena_calibration_mode = False
                    print("Arena calibration complete!")
                return  # Add this to prevent regular calibration handling
            elif detector.is_calibrating:
                if detector.add_calibration_point(x, y):
                    print("Calibration complete! Now collecting training data...")
                    print("Press 'T' when ready to train the detector")
            else:
                print(f"Clicked at ({x, y})")
                for i, points in enumerate(detector.corners):
                    for j, p in enumerate(points):
                        if np.linalg.norm(np.array(p) - np.array([x, y])) < 10:
                            detector.selected_point = (i, j)
                            print(f"Selected point: {detector.selected_point}")
                            break
        elif event == cv2.EVENT_MOUSEMOVE and detector.selected_point is not None:
            i, j = detector.selected_point
            detector.update_shared_corners(i, j, (x, y))
            print(f"Moved point {detector.selected_point} to ({x, y})")
        elif event == cv2.EVENT_LBUTTONUP:
            if detector.selected_point is not None:
                i, j = detector.selected_point
                detector.update_shared_corners(i, j, (x, y))
                detector.selected_point = None
                # print(f"Updated corners: {detector.corners}")
                # for idx, corner in enumerate(detector.corners):
                #    print(f"Square {idx}: {corner}")

    cv2.namedWindow('Chess Piece Detector')
    cv2.setMouseCallback('Chess Piece Detector', mouse_callback, detector)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam. Exiting...")
            break

        display_frame = frame.copy()
        
        # Handle Arena calibration
        if detector.arena_controller.calibrating:
            # Show instructions on frame
            cv2.putText(display_frame, "Arena Calibration Mode - Move mouse to corners and press SPACE", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Draw current mouse position
            x, y = pyautogui.position()
            cv2.putText(display_frame, f"Mouse: ({x}, {y})", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Draw already placed points
            for i, point in enumerate(detector.arena_controller.arena_corners):
                cv2.putText(display_frame, f"{i+1}: {point}", 
                           (10, 140 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if detector.is_calibrating:
            detector.draw_calibration(display_frame)
        else:
            display_frame = detector.process_frame(display_frame)
            if not detector.is_trained:
                cv2.putText(display_frame, "Model not trained", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                predictions = detector.predict(frame)
                # Optionally, draw predictions on the board
                for i in range(8):
                    for j in range(8):
                        if predictions[i][j] == 1:
                            corners = detector.corners[i * 8 + j]
                            center_x = int(sum([pt[0] for pt in corners]) / 4)
                            center_y = int(sum([pt[1] for pt in corners]) / 4)
                            cv2.putText(display_frame, "x", (center_x-8, center_y+8),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Seamless move detection based on movement
                movement = detector.detect_movement(frame)
                if movement:
                    print("Movement detected... waiting for piece to settle")
                    stable_count = 0
                    required_stable_frames = 10  # number of consecutive frames with no movement required
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if detector.detect_movement(frame):
                            stable_count = 0
                            print("Movement still detected!")
                        else:
                            stable_count += 1
                            print(f"Stable frame count: {stable_count}")
                        if stable_count >= required_stable_frames:
                            print("Movement has settled.")
                            break
                        time.sleep(0.1)  # short delay between checks
                    # Once movement has settled, re-capture board state and detect move
                    current_state = detector.predict(frame)
                    move = detector.detect_move(current_state)
                    if move:
                        print(f"Move detected: {move}")
                        time.sleep(0.5)
                        if detector.execute_move(move):
                            print("Move executed successfully")
                            time.sleep(2)  # Allow time for the move to register
                            blacks_move_info = detector.capture_black_move()
                            if blacks_move_info:
                                from_square, to_square, piece, uci_move = blacks_move_info
                                print("Black's move captured and processed")
                                print("Move info:", blacks_move_info)
                                joint_trajectory = robot_controller.generate_move_trajectory(uci_move)
                                if joint_trajectory:
                                    print("Joint trajectory generated successfully")
                                    # Display joint angles for debugging
                                    for i, joint_angles in enumerate(joint_trajectory):
                                        print(f"Point {i+1}: {np.round(joint_angles, 6)}")                               
                                    
                                    # Serialize and publish the trajectory to qnode
                                    trajectory_json = json_trajectory(joint_trajectory)
                                    print(trajectory_json)
                                    rospy.loginfo(f"Publishing trajectory with {len(joint_trajectory)} points")
                                    pub.publish(trajectory_json)
                                    print(f"Trajectory published to chess_moves topic")
                                    time.sleep(15)
                            else:
                                print("Failed to capture black's move")
                        else:
                            print("Invalid move detected")

        cv2.imshow('Chess Piece Detector', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('1'):
            if not detector.is_trained:
                print("Capturing frame with no pieces...")
                empty_board = np.zeros((8, 8), dtype=int)
                detector.collect_training_data(frame, empty_board)
        elif key == ord('2'):
            if not detector.is_trained:
                print("Capturing frame with white pieces on lower half...")
                lower_half_board = np.zeros((8, 8), dtype=int)
                lower_half_board[4:8, :] = 1
                detector.collect_training_data(frame, lower_half_board)
        elif key == ord('3'):
            if not detector.is_trained:
                print("Capturing frame with black pieces on upper half...")
                upper_half_board = np.zeros((8, 8), dtype=int)
                upper_half_board[0:4, :] = 1
                detector.collect_training_data(frame, upper_half_board)
        elif key == ord('4'):
            if not detector.is_trained:
                print("Capturing frame with normal chess start...")
                normal_start_board = np.zeros((8, 8), dtype=int)
                normal_start_board[0:2, :] = 1
                normal_start_board[6:8, :] = 1
                detector.collect_training_data(frame, normal_start_board)
        elif key == ord('t'):
            print("Training the model...")
            #print(detector.corners)
            detector.train()
            print("Model trained successfully!")
            # Update robot controller with final corner positions
            if len(detector.corners) == 64:
                print("Updating robot controller with board corners...")
                robot_controller.update_from_detector_corners(detector.corners)
            else:
                print(f"Warning: Expected 64 corners but got {len(detector.corners)}. Robot controller not updated.")
        elif key == ord('s'):
            filename = input("Enter filename to save data: ")
            detector.save_data(filename)
        elif key == ord('l'):
            filename = input("Enter filename to load data: ")
            detector.load_data(filename)
            # Get the corners (this might be needed to regenerate them from the loaded points)
            detector.get_square_points()
            # Update robot controller with loaded corner positions
            if len(detector.corners) == 64:
                print("Updating robot controller with loaded board corners...")
                robot_controller.update_from_detector_corners(detector.corners)
            else:
                print(f"Warning: Expected 64 corners but got {len(detector.corners)}. Robot controller not updated.")
        elif key == ord('a'):
            detector.arena_controller.start_calibration()
        elif key == ord(' '):  # Space key
            if detector.arena_controller.calibrating:
                if detector.arena_controller.add_corner():
                    print("Arena calibration complete!")
        elif key == 27:  # ESC key
            if detector.arena_controller.calibrating:
                detector.arena_controller.calibrating = False
                detector.arena_controller.arena_corners = []
                print("Arena calibration cancelled")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


