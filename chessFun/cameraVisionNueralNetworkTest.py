import cv2
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import chess
import pyautogui
import pyperclip
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten

class ChessGameTracker:
    def __init__(self):
        self.board = chess.Board()
        self.last_detected_state = self.get_initial_board_array()
        self.is_whites_turn = True
        self.waiting_for_move = True
        
    def get_initial_board_array(self):
        """Convert initial chess board state to 8x8 numpy array
        1 = piece present, 0 = empty square"""
        board_array = np.zeros((8, 8), dtype=int)
        # Set initial piece positions
        board_array[0:2, :] = 1  # Black pieces
        board_array[6:8, :] = 1  # White pieces
        return board_array
        
    def detect_move(self, current_state):
        """
        Detect chess move by comparing current board state with last known state
        Returns: (from_square, to_square) or None if no valid move detected
        """
        if not self.waiting_for_move:
            return None
            
        # Find changed squares
        diff = current_state != self.last_detected_state
        changed_squares = np.where(diff)
        
        # Convert to list of (row, col) coordinates
        changes = list(zip(changed_squares[0], changed_squares[1]))
        
        # For white's move, we expect movement from bottom rows (6-7) to upper rows
        if self.is_whites_turn:
            # Find all possible 'from' squares (where piece disappeared)
            from_squares = [(r, c) for r, c in changes if r >= 6 and 
                        self.last_detected_state[r][c] == 1 and 
                        current_state[r][c] == 0]
            
            # Find all possible 'to' squares (where piece appeared)
            to_squares = [(r, c) for r, c in changes if r < 6 and 
                        self.last_detected_state[r][c] == 0 and 
                        current_state[r][c] == 1]
            
        else:  # Black's move
            # For black's move, we expect movement from top rows (0-1) to lower rows
            from_squares = [(r, c) for r, c in changes if r <= 1 and 
                        self.last_detected_state[r][c] == 1 and 
                        current_state[r][c] == 0]
            
            to_squares = [(r, c) for r, c in changes if r > 1 and 
                        self.last_detected_state[r][c] == 0 and 
                        current_state[r][c] == 1]
        
        # Try all possible combinations of from and to squares
        for from_square in from_squares:
            for to_square in to_squares:
                # Convert to algebraic notation
                from_alg = self.square_to_algebraic(*from_square)
                to_alg = self.square_to_algebraic(*to_square)
                move_uci = from_alg + to_alg
                
                # Validate move
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in self.board.legal_moves:
                        return (from_square, to_square)
                except ValueError:
                    continue
        
        return None
        
    def square_to_algebraic(self, row, col):
        """Convert row,col coordinates to algebraic notation (e.g., 'e2')"""
        file = chr(ord('a') + col)
        rank = str(8 - row)
        return file + rank
        
    def update_game_state(self, move_squares):
        """
        Update game state with new move
        move_squares: tuple of ((from_row, from_col), (to_row, to_col))
        """
        from_square = self.square_to_algebraic(*move_squares[0])
        to_square = self.square_to_algebraic(*move_squares[1])
        
        # Create move in UCI format (e.g., 'e2e4')
        move_uci = from_square + to_square
        
        # Verify move is legal
        move = chess.Move.from_uci(move_uci)
        if move in self.board.legal_moves:
            self.board.push(move)
            self.is_whites_turn = not self.is_whites_turn
            self.waiting_for_move = False
            return True
            
        return False
        
    def set_opponent_move(self, move_uci):
        """Set the opponent's move from UCI format and update board state"""
        move = chess.Move.from_uci(move_uci)
        if move in self.board.legal_moves:
            self.board.push(move)
            self.is_whites_turn = not self.is_whites_turn
            # Update our last detected state based on new board position
            self.last_detected_state = self.board_to_array()
            self.waiting_for_move = True
            return True
        return False
        
    def board_to_array(self):
        """Convert current chess.Board state to numpy array"""
        board_array = np.zeros((8, 8), dtype=int)
        for i in range(8):
            for j in range(8):
                square = chess.square(j, 7-i)  # Convert to chess.Square
                if self.board.piece_at(square) is not None:
                    board_array[i][j] = 1
        return board_array


class ChessPieceDetector:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.nn_model = None  # Neural network model

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
        
    def train(self, training_stats, labels, use_nn=False):
        """Train the classifier on labeled data"""
        X = self.prepare_training_data(training_stats, labels)
        if use_nn:
            # Normalize the data for neural network
            X = np.array(X) / 255.0
            labels = to_categorical(labels)
            X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.2, random_state=42)

            # Define the neural network model
            self.nn_model = Sequential([
                Flatten(input_shape=(X.shape[1],)),
                Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(labels.shape[1], activation='softmax')
            ])

            # Compile the model
            self.nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            # Train the model
            self.nn_model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=32)
        else:
            # Scale the data for RandomForest
            X_scaled = self.scaler.fit_transform(X)
            self.classifier.fit(X_scaled, labels)
        
        self.is_trained = True
        
    def predict(self, stats, use_nn=False):
        """Predict if a square contains a piece"""
        if not self.is_trained:
            raise RuntimeError("Detector needs to be trained first")
        
        features = self.prepare_training_data([stats], None)[0].reshape(1, -1)
        
        if use_nn:
            # Normalize and reshape the features for neural network
            features = np.array(features) / 255.0
            prediction = self.nn_model.predict(features)
            predicted_label = np.argmax(prediction)
        else:
            # Scale the features for RandomForest
            features_scaled = self.scaler.transform(features)
            predicted_label = self.classifier.predict(features_scaled)[0]
        
        return predicted_label

    def save_nn_model(self, model_path):
        """Save the neural network model to a file"""
        if self.nn_model:
            self.nn_model.save(model_path)

    def load_nn_model(self, model_path):
        """Load the neural network model from a file"""
        self.nn_model = load_model(model_path)
        self.is_trained = True

class ChessVision:
    def __init__(self):
        self.points = []
        self.h_lines = None
        self.v_lines = None
        self.selected_square = None
        self.piece_detector = ChessPieceDetector()
        self.training_data = []
        self.is_collecting_data = False
        self.is_calibrating = True
        self.calibration_complete = False
        self.game_tracker = ChessGameTracker()
        self.last_processed_state = None
        self.move_confidence_counter = 0
        self.last_frame = None
        self.movement_threshold = 3000000  # Adjust this value based on testing
        self.movement_cooldown = 0
        self.stable_state_counter = 0
        self.last_stable_state = None
        self.pre_move_state = None  # Track the state before the move
        self.post_move_state = None  # Track the state after the move
        self.black_move_counter = 0

    def detect_board_changes(self, current_state):
        """
        Compare current board state with last stable state to detect changes
        Returns: List of changed squares [(row, col, old_value, new_value)]
        """
        if self.pre_move_state is None:
            return []
            
        changes = []
        for i in range(8):
            for j in range(8):
                if current_state[i][j] != self.pre_move_state[i][j]:
                    changes.append((i, j, 
                                 self.pre_move_state[i][j],
                                 current_state[i][j]))
        return changes

    def analyze_white_move(self, changes):
        """
        Analyze changes to detect a valid white move
        Returns: (from_square, to_square) or None if no valid move detected
        """
        if not changes:
            return None

        # For white's move, we expect:
        # 1. A piece disappearing from ranks 1-2 (board rows 6-7)
        # 2. A piece appearing on a previously empty square
        possible_moves = []

        if isinstance(changes[0], tuple) and len(changes[0]) == 2:
            # Handle output from detect_move
            from_square, to_square = changes
            possible_moves.append((from_square, to_square))
        else:
            # Handle output from detect_board_changes
            for i, (row1, col1, old_val1, new_val1) in enumerate(changes):
                for j, (row2, col2, old_val2, new_val2) in enumerate(changes):
                    if i != j:
                        if old_val1 == 1 and new_val1 == 0 and row1 >= 6:  # Piece disappeared from white's side
                            if old_val2 == 0 and new_val2 == 1:  # Piece appeared
                                from_square = (row1, col1)
                                to_square = (row2, col2)
                                possible_moves.append((from_square, to_square))

        # Validate possible moves using chess logic
        for from_square, to_square in possible_moves:
            from_square_alg = self.game_tracker.square_to_algebraic(*from_square)
            to_square_alg = self.game_tracker.square_to_algebraic(*to_square)
            move_uci = from_square_alg + to_square_alg
            move = chess.Move.from_uci(move_uci)
            if move in self.game_tracker.board.legal_moves:
                return (from_square, to_square)

        return None


    def analyze_black_move(self):
        """
        Analyze changes to detect a valid black move
        Returns: (from_square, to_square) or None if no valid move detected
        """
        #if not changes:
        #    return None

        # Scrape the move from the clipboard
        clipboard_content = pyperclip.paste()
        print(f"Clipboard content: {clipboard_content}")  # Debugging line
        print("fart")
        # Extract the move from the clipboard content
        move_str = self.extract_move_from_clipboard(clipboard_content)
        if not move_str:
            return None
        return move_str
        # # Convert the move to from_square and to_square
        # from_square_alg = move_str[:2]
        # to_square_alg = move_str[2:]
        # from_square = self.game_tracker.square_to_algebraic(from_square_alg)
        # to_square = self.game_tracker.square_to_algebraic(to_square_alg)

        # # Validate the move
        # move_uci = from_square_alg + to_square_alg
        # move = chess.Move.from_uci(move_uci)
        # if move in self.game_tracker.board.legal_moves:
        #     return (from_square, to_square)

        #return None

    def extract_move_from_clipboard(self, clipboard_content):
        """
        Extract the move from the clipboard content.
        :param clipboard_content: The content of the clipboard.
        :return: The move string in UCI format or None if not found.
        """
        self.black_move_counter +=1
        # Example logic to extract the move from the clipboard content
        # This assumes the move is the last move in the clipboard content
        lines = clipboard_content.split('\n')
        for line in reversed(lines):
            if line.startswith('1.'):
                parts = line.split()
                print(f"Extracted parts: {parts}")  # Debugging line
                if len(parts) >= 2:
                    return parts[2*self.black_move_counter]  # Assuming the move is the second part
        
        return None


    def start_calibration(self):
        """Reset calibration points"""
        self.points = []
        self.is_calibrating = True
        self.calibration_complete = False
        
    def add_calibration_point(self, x, y):
        """Add a calibration point and check if calibration is complete"""
        if len(self.points) < 4:
            self.points.append((x, y))
            
        if len(self.points) == 4:
            self.calibration_complete = True
            self.is_calibrating = False
            self.is_collecting_data = True
       
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



    def detect_movement(self, frame):
        """Detect if there's significant movement in the frame"""
        if self.last_frame is None:
            self.last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame_diff = cv2.absdiff(gray, self.last_frame)
        
        self.last_frame = gray

        movement = np.sum(frame_diff) > self.movement_threshold
        #print(np.sum(frame_diff))
        cv2.putText(frame, str(movement), (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return movement


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

                # Evaluate training accuracy
        predictions = self.piece_detector.predict(training_stats)
        accuracy = np.mean(predictions == labels) * 100
        print(f"Training Accuracy: {accuracy:.2f}%")
        
        self.is_collecting_data = False

        # Assuming training_stats is your feature data and labels are your target labels
        # Normalize the data
        training_stats = np.array(training_stats) / 255.0
        labels = np.array(labels)

        # One-hot encode the labels if they are categorical
        labels = to_categorical(labels)

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(training_stats, labels, test_size=0.2, random_state=42)

        # Define the model
        model = Sequential([
            Flatten(input_shape=(training_stats.shape[1],)),  # Adjust input shape as needed
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(labels.shape[1], activation='softmax')  # Adjust output layer size as needed
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=32)

        # Evaluate the model on the training data
        train_loss, train_accuracy = model.evaluate(X_train, y_train)
        print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

        # Evaluate the model on the validation data
        val_loss, val_accuracy = model.evaluate(X_val, y_val)
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    
    def process_frame(self, frame):
        # Detect movement first
        square_stats = []
        movement = self.detect_movement(frame)
        if movement:
            if self.pre_move_state is None:
                self.pre_move_state = self.get_current_board_state(frame)
            self.stable_state_counter = 0
            self.post_move_state = self.get_current_board_state(frame)
            return frame, square_stats
        else:
            h_lines, v_lines = self.get_square_points()
            current_state = np.zeros((8, 8), dtype=int)
            
            # Process each square (existing square analysis code)
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
                    # Update current state based on piece detection
                    is_empty = self.is_empty_square(stats)
                    current_state[i][j] = 0 if is_empty else 1
        
        # Handle movement and state changes
        if movement:
            self.stable_state_counter = 0
        else:
            
            if self.is_collecting_data == False:
                self.stable_state_counter += 1
                print(self.stable_state_counter)

                # After no movement for several frames, analyze for changes
                if self.stable_state_counter >= 1:  # Adjust this value based on testing
                    
                    if self.game_tracker.is_whites_turn and self.last_stable_state is not None:
                        changes = self.game_tracker.detect_move(current_state)
                        print(f"Detected changes: {changes}")  # Debugging line
                        move = self.analyze_white_move(changes)
                        print(f"Analyzed move: {move}")  # Debugging line
                        if move:
                            from_square = self.game_tracker.square_to_algebraic(*move[0])
                            to_square = self.game_tracker.square_to_algebraic(*move[1])
                            print(f"Detected white's move: {from_square}{to_square}")
                            
                            # Update game state
                            if self.game_tracker.update_game_state(move):
                                print("Move validated and board updated")
                                self.last_stable_state = current_state.copy()
                                pyautogui.PAUSE = 1.0
                                move_str = from_square + to_square
                                pyautogui.write(move_str)
                                # Press enter to submit
                                pyautogui.press('enter')
                                # Press F5 to copy the moves to the clipboard
                                pyautogui.PAUSE = 5.0
                                pyautogui.press('f5')
                                # Small delay to let the move register
                                time.sleep(0.5)
                                #self.is_whites_turn == False
                    
                    if not self.game_tracker.is_whites_turn and self.last_stable_state is not None:
                        print("blackTURN")
                        changes = self.game_tracker.detect_move(current_state)
                        #print(f"Detected changes: {changes}")  # Debugging line
                        move = self.analyze_black_move()
                        print(f"Analyzed move: {move}")  # Debugging line
                        if move:
                            #from_square = self.game_tracker.square_to_algebraic(*move[0])
                           # to_square = self.game_tracker.square_to_algebraic(*move[1])
                            #print(f"Detected blacks's move: {from_square}{to_square}")
                            self.game_tracker.is_whites_turn = True
                        



                    # Update stable state
                    self.last_stable_state = current_state.copy()
                    self.pre_move_state = None  # Reset pre_move_state after processing
            
        # Drawing code remains the same...
        return frame, square_stats
    
    def get_current_board_state(self, frame):
        """Get the current board state from the frame"""
        h_lines, v_lines = self.get_square_points()
        current_state = np.zeros((8, 8), dtype=int)
        
        for i in range(8):
            for j in range(8):
                corners = [
                    h_lines[i][j],
                    h_lines[i][j+1],
                    h_lines[i+1][j+1],
                    h_lines[i+1][j]
                ]
                
                stats = self.analyze_square(frame, corners, i, j)
                is_empty = self.is_empty_square(stats)
                current_state[i][j] = 0 if is_empty else 1
        
        return current_state
    
    
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
    frame_count = 0
    print("Chess Vision Analysis started.")
    print("Please calibrate the board first:")
    print("1. Click the top-left corner")
    print("2. Click the top-right corner")
    print("3. Click the bottom-right corner")
    print("4. Click the bottom-left corner")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if chess_vision.is_calibrating:
            chess_vision.draw_calibration(frame)
            cv2.imshow('Chess Vision', frame)
        else:
            processed_frame, stats = chess_vision.process_frame(frame)
            
            # Add game state info to frame
            status = "White's turn" if chess_vision.game_tracker.is_whites_turn else "Black's turn"
            cv2.putText(processed_frame, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Chess Vision', processed_frame)
            
            # Handle square analysis
            if chess_vision.selected_square is not None:
                for stat in stats:
                    if stat['pos'] == f"{chess_vision.selected_square[0]},{chess_vision.selected_square[1]}":
                        print("\nSquare Analysis:")
                        print(f"Position: {stat['pos']}")
                        print(f"HSV Means - H: {stat['hsv_hue_mean']:.1f}, "
                              f"S: {stat['hsv_sat_mean']:.1f}, V: {stat['hsv_val_mean']:.1f}")
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
        elif key == ord('m'):
            # Example of setting opponent's move
            move = input("Enter opponent's move (e.g., e7e5): ")
            if chess_vision.game_tracker.set_opponent_move(move):
                print("Opponent's move set successfully")
            else:
                print("Invalid move")


         # Collect training data every N frames after calibration
        if chess_vision.is_collecting_data and not chess_vision.piece_detector.is_trained:
            frame_count += 1
            if frame_count >= 30:
                frame_count = 0
                print("Collected data from frame")
    
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_webcam()