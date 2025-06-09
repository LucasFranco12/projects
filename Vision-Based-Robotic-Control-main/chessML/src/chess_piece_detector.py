# Chess Piece Detector
# This class handles piece prescence detection and move execution on the chess board.
# This is through ML models and computer vision techniques.
# The piece prescence prediction is then compared to the interal games state of our chess game to recognize player move

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
import json

from chess_arena_controller import ChessArenaController


class ChessPieceDetector:
    def __init__(self):
        # Initialize variables for model, training data, and calibration
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
        
        # Folder to save training data samples
        self.training_folder = "/home/l/catkin_ws/src/chessML/src/training_samples"
        if not os.path.exists(self.training_folder):
            os.makedirs(self.training_folder)
        
        # Initialize arena controller
        self.arena_controller = ChessArenaController()
        self.arena_calibration_mode = False

    # Collect training data from a frame and save it
    def collect_training_data(self, frame, label):
        h_lines, v_lines, corners = self.get_square_points()
        
        # Add rotation variations to augment data
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
                    
                    # Save the training sample to disk
                    sample = {'features': stats, 'label': label[i][j]}
                    sample_filename = os.path.join(self.training_folder, f"sample_{int(time.time()*1000)}_{i}_{j}_{angle}.pkl")
                    with open(sample_filename, 'wb') as f:
                        pickle.dump(sample, f)

    # Rotate a frame by a given angle
    def _rotate_frame(self, frame, angle):
        height, width = frame.shape[:2]
        center = (width/2, height/2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(frame, rotation_matrix, (width, height))

    # Rotate corner points to match the rotated frame
    def _rotate_corners(self, corners, angle, shape):
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

    # Analyze a square to extract features
    def analyze_square(self, frame, square_corners):
        # Convert corners to tuples of integers
        corners = [tuple(map(int, corner)) for corner in square_corners]
        
        # Create a mask of the same size as the frame, initialized to zero
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Fill the mask with a polygon defined by the square corners
        cv2.fillPoly(mask, [np.array(corners)], 255)
        
        # Apply the mask to the frame to isolate the region of interest (ROI)
        roi = cv2.bitwise_and(frame, frame, mask=mask)

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

    # Train the model using collected training data
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

    # Predict the state of the chessboard from a frame
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

    # Get the points of the squares on the chessboard
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

    # Start the calibration process
    def start_calibration(self):
        self.points = []
        self.is_calibrating = True
        self.calibration_complete = False

    # Add a calibration point
    def add_calibration_point(self, x, y):
        if len(self.points) < 4:
            self.points.append((x, y))
            
        if len(self.points) == 4:
            self.calibration_complete = True
            self.is_calibrating = False
            return True
        return False

    # Draw calibration points and lines on the frame
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

    # Process a frame to detect chessboard squares
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

    # Save data (model, training data, etc.) to a file
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

    # Update shared corners of squares
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

    # Load data (model, training data, etc.) from a file
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

    # Convert the chessboard state to a numpy array
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

    # Detect a move by comparing the current state with the internal game state
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

    # Get the value of a piece based on its type and position
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

    # Convert a chess.Board to a numpy array
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

    # Convert row, col coordinates to algebraic notation
    def square_to_algebraic(self, row, col):
        #Convert row,col coordinates to algebraic notation
        file = chr(ord('a') + col)
        rank = str(8 - row)
        return file + rank

    # Execute a detected move using the Arena GUI
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

    # Capture black's move from the clipboard
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

    # Extract black's move from clipboard content
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

    # Convert algebraic notation to UCI format
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