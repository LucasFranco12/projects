# Application that trains a random forest classifier to detect chess pieces on a chess board....

import cv2
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import chess
import pyautogui
import pyperclip
from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt

class ChessGameTracker:
    def __init__(self):
        self.board = chess.Board()
        self.is_whites_turn = True
        self.waiting_for_move = True

    # Detect chess move by comparing current board state with last known state Returns: (from_square, to_square) or None if no valid move detected
    def detect_move(self, current_state):

        # Find changed squares
        print("current_state", current_state)
        print("self.current_state", self.board.fen())
        diff = current_state != self.board_to_array()
        changed_squares = np.where(diff)
        
        # Convert to list of (row, col) coordinates
        changes = list(zip(changed_squares[0], changed_squares[1]))
        print("Detected changes:", changes) 
        
        # For white's move, we expect movement from bottom rows (6-7) to upper rows
        if self.is_whites_turn:
            # Find all possible 'from' squares (where piece disappeared)
            from_squares = [(r, c) for r, c in changes if r >= 6 and 
                        self.board.piece_at(chess.square(c, 7-r)) is not None and 
                        current_state[r][c] == 0]
            
            # Find all possible 'to' squares (where piece appeared)
            to_squares = [(r, c) for r, c in changes if r < 6 and 
                        self.board.piece_at(chess.square(c, 7-r)) is None and 
                        current_state[r][c] == 1]
            
        else:  # Black's move
            # For black's move, we expect movement from top rows (0-1) to lower rows
            from_squares = [(r, c) for r, c in changes if r <= 1 and 
                        self.board.piece_at(chess.square(c, 7-r)) is not None and 
                        current_state[r][c] == 0]
            
            to_squares = [(r, c) for r, c in changes if r > 1 and 
                        self.board.piece_at(chess.square(c, 7-r)) is None and 
                        current_state[r][c] == 1]
        
        print("From squares:", from_squares)  
        print("To squares:", to_squares)  
        
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
                        print(f"Detected move: {move_uci}")  
                        return (from_square, to_square)
                except ValueError:
                    continue
        
        return None

    # Convert row,col coordinates to algebraic notation (e.g., 'e2')        
    def square_to_algebraic(self, row, col):

        file = chr(ord('a') + col)
        rank = str(8 - row)
        return file + rank

    # Update game state with new move move_squares: tuple of ((from_row, from_col), (to_row, to_col))   
    def update_game_state(self, move_squares):
        
        if move_squares[0] is None:
            print("Error: from_square is None")
            return False
        
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
            print("Updated game state")  
            print(self.board)  
            return True
            
        return False

    # Update game state with new move move_squares: tuple of ((from_row, from_col), (to_row, to_col))
    def update_game_state_black(self, move_squares):
      
        from_square = move_squares[0]
        to_square = move_squares[1]

        # Create move in UCI format (e.g., 'e2e4')
        move_uci = from_square + to_square
        
        # Verify move is legal
        move = chess.Move.from_uci(move_uci)
        if move in self.board.legal_moves:
            self.board.push(move)
            self.is_whites_turn = not self.is_whites_turn
            self.waiting_for_move = False
            print("Updated game state")  
            print(self.board)  
            return True
            
        return False

    # Convert current chess.Board state to numpy array   
    def board_to_array(self):
       
        board_array = np.zeros((8, 8), dtype=int)
        for i in range(8):
            for j in range(8):
                square = chess.square(j, 7-i)  # Convert to chess.Square
                if self.board.piece_at(square) is not None:
                    board_array[i][j] = 1
        return board_array

# This model trains a random forest classifier to detect chess pieces on a chess board
# It uses color statistics and edge detection features to classify squares as empty or occupied
# The labeled training data is normal setup of chess board with pieces on it
class ChessPieceDetector:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=350, random_state=42, max_depth=None,min_samples_split=2, min_samples_leaf=1,bootstrap=True,criterion='gini')
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_training_data(self, stats_list, labels):
        # Convert stats dictionaries to feature matrix
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
                stats['gray_std'],
                stats['edge_mean'],
                stats['edge_std'],
                *stats['color_histogram']
            ]
            feature_vector.extend(stats['binary_center_region'])
            features.append(feature_vector)
        return np.array(features)

    # Train the classifier on labeled data   
    def train(self, training_stats, labels):
       
        X = self.prepare_training_data(training_stats, labels)
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, labels)
        self.is_trained = True
        
    def predict(self, stats):
        # Predict if a square contains a piece
        if not self.is_trained:
            raise RuntimeError("Detector needs to be trained first")
            
        features = self.prepare_training_data([stats], None)[0].reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        return self.classifier.predict(features_scaled)[0]
    

    # Unncomment to see the tree structure
    # def print_tree(self, tree_index=0):
    #     if not self.is_trained:
    #         raise RuntimeError("Detector needs to be trained first")
        
    #     tree = self.classifier.estimators_[tree_index]
    #     tree_text = export_text(tree, feature_names=[
    #         'hsv_hue_mean', 'hsv_sat_mean', 'hsv_val_mean', 
    #         'hsv_hue_std', 'hsv_sat_std', 'hsv_val_std', 
    #         'gray_mean', 'gray_std', 'edge_mean', 'edge_std'
    #     ] + [f'binary_center_region_{i}' for i in range(400)])
    #     print(tree_text)
    # #Visualize a decision tree
    # def plot_tree(self, tree_index=0):
    #     
    #     if not self.is_trained:
    #         raise RuntimeError("Detector needs to be trained first")
        
    #     tree = self.classifier.estimators_[tree_index]
    #     plt.figure(figsize=(20, 10))
    #     plot_tree(tree, feature_names=[
    #         'hsv_hue_mean', 'hsv_sat_mean', 'hsv_val_mean', 
    #         'hsv_hue_std', 'hsv_sat_std', 'hsv_val_std', 
    #         'gray_mean', 'gray_std', 'edge_mean', 'edge_std'
    #     ] + [f'binary_center_region_{i}' for i in range(400)], filled=True)
    #     plt.show()
    

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
        self.move_confidence_counter = 0
        self.last_frame = None
        self.movement_threshold = 10000000  # Adjust this value based on your camera and lighting
        self.stable_state_counter = 0
        self.post_move_state = None  
        self.black_move_counter = 1
        self.cooldown_period = 2  

    # Analyze changes to detect a valid white move Returns: (from_square, to_square) or None if no valid move detected        
    def analyze_white_move(self, changes):

        if not changes:
            return None

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

    # Analyze changes to detect a valid black moveReturns: (from_square, to_square) or None if no valid move detected this function is being built as testing goes further
    def analyze_black_move(self):
        
        # Scrape the move from the clipboard
        clipboard_content = pyperclip.paste()
        print(f"Clipboard content: {clipboard_content}") 
        
        # Extract the move from the clipboard content
        move_str = self.extract_move_from_clipboard(clipboard_content)
        if not move_str:
            return None
        
        print(f"Extracted move: {move_str}")  
        
        # Convert the move to from_square and to_square
        if len(move_str) == 2:
            # Handle initial moves like 'e5'
            to_square_alg = move_str
            to_square = chess.square(ord(to_square_alg[0]) - ord('a'), int(to_square_alg[1]) - 1)
            from_square = self.find_pawn_from_square(to_square)
        elif len(move_str) == 4:
            # Handle moves like 'e2e4'
            from_square_alg = move_str[:2]
            to_square_alg = move_str[2:]
            print(f"From square: {from_square_alg}, To square: {to_square_alg}")  
            from_square = chess.square(ord(to_square_alg[0]) - ord('a'), int(to_square_alg[1]) - 1)
            to_square = chess.square(ord(to_square_alg[0]) - ord('a'), int(to_square_alg[1]) - 1)
        elif len(move_str) == 3:
            # Handle moves like 'Nf3' or 'Nf6'
            piece = move_str[0]
            to_square_alg = move_str[1:]
            to_square = chess.square(ord(to_square_alg[0]) - ord('a'), int(to_square_alg[1]) - 1)
            from_square = self.find_from_square_by_piece(piece, to_square)
        elif 'x' in move_str:
            # Handle capture moves like 'Nxe4'
            piece = move_str[0]
            to_square_alg = move_str[2:]
            to_square = chess.square(ord(to_square_alg[0]) - ord('a'), int(to_square_alg[1]) - 1)
            from_square = self.find_from_square_by_piece(piece, to_square)
        else: 
            #handle move like Ng1-f3
            piece = move_str[0]
            from_square_alg = move_str[1:3]
            to_square_alg = move_str[4:]
            from_square = chess.square(ord(from_square_alg[0]) - ord('a'), int(from_square_alg[1]) - 1)
            to_square = chess.square(ord(to_square_alg[0]) - ord('a'), int(to_square_alg[1]) - 1)

        if from_square is None:
            print(f"Error: from_square not found for move {move_str}")
            return None, to_square

        from_square_alg = chess.square_name(from_square)
        to_square_alg = chess.square_name(to_square)
        print(f"From square: {from_square_alg}, To square: {to_square_alg}")  
        return (from_square_alg, to_square_alg)


    # Find the from_square for a pawn move based on the to_square.
    def find_pawn_from_square(self, to_square):
       
        row, col = chess.square_rank(to_square), chess.square_file(to_square)
        if self.game_tracker.is_whites_turn:
            from_row = row - 1
            double_move_row = row - 2
            pawn_color = chess.WHITE
        else:
            from_row = row + 1
            double_move_row = row + 2
            pawn_color = chess.BLACK
    
        from_square = chess.square(col, from_row)
        piece = self.game_tracker.board.piece_at(from_square)
        if piece is not None and piece.piece_type == chess.PAWN and piece.color == pawn_color:
            return from_square
    
        # Handle double move for pawns
        from_square = chess.square(col, double_move_row)
        piece = self.game_tracker.board.piece_at(from_square)
        if piece is not None and piece.piece_type == chess.PAWN and piece.color == pawn_color:
            print(f"Finding pawn from_square: {chess.square_name(from_square)} for to_square: {chess.square_name(to_square)}") 
            return from_square
    
        return None
    
    # Find the from_square based on the to_square and the current board state.
    def find_from_square(self, to_square):
        
        piece = self.game_tracker.board.piece_at(to_square)
        if piece is None:
            print(f"No piece found at to_square: {chess.square_name(to_square)}") 
            return None
    
        print(f"Piece at to_square {chess.square_name(to_square)}: {piece}")  
    
        for move in self.game_tracker.board.legal_moves:
            if move.to_square == to_square:
                print(f"Found legal move from {chess.square_name(move.from_square)} to {chess.square_name(to_square)}")  
                return move.from_square
    
        print(f"No legal move found to to_square: {chess.square_name(to_square)}") 
        return None

    #Find the from_square based on the piece type and the to_square.
    def find_from_square_by_piece(self, piece, to_square):
        
        for move in self.game_tracker.board.legal_moves:
            if move.to_square == to_square and self.game_tracker.board.piece_at(move.from_square).symbol().upper() == piece:
                return move.from_square

        return None

    #  Extract the move from the clipboard content.
    def extract_move_from_clipboard(self, clipboard_content):
  
        # Blacks move is always the second item in the array after the move number
        lines = clipboard_content.split('\n')
        move_number = f"{self.black_move_counter}."
        print(f"Looking for move number: {move_number}")  
        for line in lines:
            if move_number in line:
                parts = line.split(move_number)
                if len(parts) > 1:
                    moves = parts[1].strip().split()
                    if len(moves) >= 2:
                        move = moves[1].strip('[]"')  
                        self.black_move_counter += 1  
                        print(f"Extracted move: {move}")  
                        return move  # Extract black's move
        print("No move found")  
        return None


    # Board calibration
    def start_calibration(self):
        
        self.points = []
        self.is_calibrating = True
        self.calibration_complete = False

    # Add a calibration point and check if calibration is complete    
    def add_calibration_point(self, x, y):
       
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

    # Draw calibration points and lines
    def draw_calibration(self, frame):
        
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

    # Initialize the board state with starting position
    def get_initial_board_state(self):
   
        board = np.zeros((8, 8), dtype=int)
        board[0:2, :] = 1  # Black pieces
        board[6:8, :] = 1  # White pieces
        return board
    
    # Get the grid points with perspective correction
    def get_square_points(self):
       
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


    # Detect movement in the frame
    def detect_movement(self, frame):

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

    # Analyze the square and return detailed color statistics
    def analyze_square(self, frame, square_corners, row, col):
      
        corners = [tuple(map(int, corner)) for corner in square_corners]
        
        # Create a mask for the square
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(corners)], 255)
        
        # Extract the square region
        roi = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Convert the square region to HSV and grayscale
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate mean and standard deviation of HSV and grayscale values
        hsv_mean = cv2.mean(hsv_roi, mask=mask)
        hsv_std = cv2.meanStdDev(hsv_roi, mask=mask)[1]
        gray_mean = cv2.mean(gray_roi, mask=mask)[0]
        gray_std = cv2.meanStdDev(gray_roi, mask=mask)[1][0][0]

        # Extract the center 20x20 pixel region
        x, y, w, h = cv2.boundingRect(np.array(corners))
        center_x, center_y = x + w // 2, y + h // 2
        half_size = 10  # Half of 20 pixels
        center_region = gray_roi[center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size]
    
        # Convert the center region to binary (0 or 1) based on a threshold
        _, binary_center_region = cv2.threshold(center_region, 127, 1, cv2.THRESH_BINARY)
    
        # Flatten the binary region and convert to a list
        binary_center_region_flat = binary_center_region.flatten().tolist()

        # Calculate edge detection features
        edges = cv2.Canny(gray_roi, 100, 200)
        edge_mean = np.mean(edges)
        edge_std = np.std(edges)

        # Calculate color histogram features
        hist = cv2.calcHist([hsv_roi], [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        stats = {
            'pos': f"{row},{col}",
            'hsv_hue_mean': hsv_mean[0],
            'hsv_sat_mean': hsv_mean[1],
            'hsv_val_mean': hsv_mean[2],
            'hsv_hue_std': hsv_std[0][0],
            'hsv_sat_std': hsv_std[1][0],
            'hsv_val_std': hsv_std[2][0],
            'gray_mean': gray_mean,
            'gray_std': gray_std,
            'binary_center_region': binary_center_region_flat,
            'edge_mean': edge_mean,
            'edge_std': edge_std,
            'color_histogram': hist
        }
        
        return stats
    
    # Detect if square is empty using trained classifier
    def is_empty_square(self, stats):
        if not self.piece_detector.is_trained:
            # Default to simple threshold-based detection during data collection
            # This was taken from printing one of our trees
            if stats['hsv_val_mean'] <= 1.06:
                if stats['gray_mean'] <= -0.68:
                    if stats['hsv_hue_std'] <= 0.75:
                        if stats['hsv_val_mean'] <= -1.14:
                            return 0.0
                        else:
                            if stats['hsv_sat_mean'] <= -0.89:
                                if stats['hsv_val_std'] <= 0.52:
                                    if stats['hsv_val_mean'] <= -0.86:
                                        if stats['hsv_hue_std'] <= 0.43:
                                            if stats['hsv_val_std'] <= -0.30:
                                                return 1.0
                                            else:
                                                return 0.0
                                        else:
                                            return 0.0
                                    else:
                                        if stats['gray_std'] <= -1.38:
                                            return 1.0
                                        else:
                                            return 0.0
                                else:
                                    return 1.0
                            else:
                                if stats['hsv_val_mean'] <= -0.84:
                                    return 1.0
                                else:
                                    return 0.0
                    else:
                        if stats['hsv_hue_std'] <= 1.01:
                            if stats['gray_mean'] <= -1.12:
                                return 0.0
                            else:
                                if stats['hsv_hue_std'] <= 1.21:
                                    if stats['hsv_val_std'] <= 0.52:
                                        if stats['hsv_val_mean'] <= -0.88:
                                            return 1.0
                                        else:
                                            return 0.0
                                    else:
                                        return 1.0
                                else:
                                    return 0.0
                        else:
                            if stats['gray_mean'] <= -1.18:
                                return 0.0
                            else:
                                return 1.0
                else:
                    if stats['gray_std'] <= -1.73:
                        return 1.0
                    else:
                        if stats['hsv_hue_mean'] <= -0.83:
                            if stats['hsv_hue_mean'] <= -0.91:
                                if stats['hsv_hue_std'] <= -1.14:
                                    if stats['hsv_val_std'] <= 0.54:
                                        return 0.0
                                    else:
                                        return 1.0
                                else:
                                    return 0.0
                            else:
                                if stats['hsv_sat_std'] <= 0.31:
                                    return 0.0
                                else:
                                    return 1.0
                        else:
                            return 0.0
            else:
                if stats['hsv_val_mean'] <= 1.11:
                    if stats['hsv_sat_std'] <= -0.92:
                        return 1.0
                    else:
                        return 0.0
                else:
                    return 1.0
        else:
            return self.piece_detector.predict(stats)
        

    # Collect training data for the piece detector    
    def train_detector(self):
       
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
        #self.piece_detector.print_tree(tree_index=0)  # Print the first tree
        #self.piece_detector.plot_tree(tree_index=0)   # Visualize the first tree

        # Evaluate training accuracy
        X = self.piece_detector.prepare_training_data(training_stats, labels)
        X_scaled = self.piece_detector.scaler.transform(X)
        predictions = self.piece_detector.classifier.predict(X_scaled)
        accuracy = np.mean(predictions == labels) * 100
        print(f"Training Accuracy: {accuracy:.2f}%")
        
        self.is_collecting_data = False
    
    # Process the frame and detect chess pieces
    def process_frame(self, frame):
        # Detect movement first
        square_stats = []
        movement = self.detect_movement(frame)
        if movement:
            self.stable_state_counter = 0
            return frame, square_stats
        else:
            h_lines, v_lines = self.get_square_points()
            current_state = np.zeros((8, 8), dtype=int)
            
            # Process each square in the grid
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
                    current_state[i][j] = 0 if is_empty else 1
        
        # Handle movement and state changes
        if movement:
            self.stable_state_counter = 0
        else:
            self.stable_state_counter += 1
            print(self.stable_state_counter)

            # After no movement for several frames capture final state
            if self.stable_state_counter >= self.cooldown_period and not self.is_collecting_data:
                self.post_move_state = self.get_current_board_state(frame)
                print("Captured final state") 

                print(self.post_move_state)
                self.stable_state_counter = 0  
                
                if self.game_tracker.is_whites_turn:
                    changes = self.game_tracker.detect_move(self.post_move_state)
                    print(f"Detected changes: {changes}") 
                    move = self.analyze_white_move(changes)
                    print(f"Analyzed move: {move}") 
                    if move:
                        from_square = self.game_tracker.square_to_algebraic(*move[0])
                        to_square = self.game_tracker.square_to_algebraic(*move[1])
                        print(f"Detected white's move: {from_square}{to_square}")
   
                        # Update game state
                        if self.game_tracker.update_game_state(move):
                            print("Move validated and board updated")
                 
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
                            
                
                if not self.game_tracker.is_whites_turn:
                    print("blackTURN")
                    changes = self.game_tracker.detect_move(self.post_move_state)
                    move = self.analyze_black_move()
                    print(f"Analyzed move: {move}")  # Debugging line
                    if self.game_tracker.update_game_state_black(move):
                        #update current board to reflect changed board here 
                        
                        self.game_tracker.is_whites_turn = True
                        print(self.game_tracker.is_whites_turn)
                        #self.stable_state_counter = 0  # Reset cooldown period

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
    
# Handle mouse clicks for both calibration and square selection
def mouse_clicky(event, x, y, flags, param):

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


# Test the webcam with ChessVision
def test_webcam():
    cap = cv2.VideoCapture(0)
    
    # Set the resolution to 1080p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 810)
    
    chess_vision = ChessVision()
    
    cv2.namedWindow('Chess Vision')
    cv2.setMouseCallback('Chess Vision', mouse_clicky, chess_vision)
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

