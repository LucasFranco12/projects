import cv2
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import chess


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
        
        # We expect exactly two changes for a normal move
        if len(changed_squares[0]) != 2:
            return None
            
        # Convert to list of (row, col) coordinates
        changes = list(zip(changed_squares[0], changed_squares[1]))
        
        # For white's move, we expect movement from bottom rows (6-7) to upper rows
        if self.is_whites_turn:
            # Find 'from' square (where piece disappeared)
            from_squares = [(r, c) for r, c in changes if r >= 6 and 
                          self.last_detected_state[r][c] == 1 and 
                          current_state[r][c] == 0]
            
            # Find 'to' square (where piece appeared)
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
        
        if len(from_squares) == 1 and len(to_squares) == 1:
            return (from_squares[0], to_squares[0])
            
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
        
        self.movement_threshold = 800000  # Adjust this value based on testing
        self.movement_cooldown = 0
        self.stable_state_counter = 0
        self.last_stable_state = None
        self.initial_frame = None
        self.final_frame = None
        self.move_in_progress = False
        self.last_frame = None
        


    def draw_board_grid(self, frame, h_lines, v_lines):
        """Draw chess board grid with rank and file labels in correct positions"""
        # Draw horizontal lines
        for i in range(9):
            points = np.array([h_lines[i][j] for j in range(9)], dtype=np.int32)
            cv2.polylines(frame, [points.reshape(-1, 1, 2)], False, (0, 255, 0), 2)
            
            # Add rank labels (8-1) on the left side
            if i < 8:
                label_point = (int(h_lines[i][0][0]) - 20, int(h_lines[i][0][1]) + 20)
                cv2.putText(frame, str(8-i), label_point,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw vertical lines
        for j in range(9):
            points = np.array([v_lines[i][j] for i in range(9)], dtype=np.int32)
            cv2.polylines(frame, [points.reshape(-1, 1, 2)], False, (0, 255, 0), 2)
            
            # Add file labels (a-h) at the bottom
            if j < 8:
                # Calculate position under the board using the bottom grid points
                bottom_y = int(h_lines[8][j][1])  # Get y-coordinate of bottom grid line
                x_pos = int((h_lines[8][j][0] + h_lines[8][j+1][0]) / 2)  # Center between grid points
                label_point = (x_pos, bottom_y + 25)  # Offset by 25 pixels below the board
                cv2.putText(frame, chr(ord('a') + j), label_point,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # If a square is selected, highlight it
        if self.selected_square is not None:
            i, j = self.selected_square
            corners = np.array([
                h_lines[i][j],
                h_lines[i][j+1],
                h_lines[i+1][j+1],
                h_lines[i+1][j]
            ], dtype=np.int32)
            cv2.polylines(frame, [corners.reshape(-1, 1, 2)], True, (0, 0, 255), 2)



    def detect_board_changes(self, current_state):
        """
        Compare current board state with last stable state to detect changes
        Returns: List of changed squares [(row, col, old_value, new_value)]
        """
        if self.last_stable_state is None:
            return []
            
        changes = []
        for i in range(8):
            for j in range(8):
                if current_state[i][j] != self.last_stable_state[i][j]:
                    changes.append((i, j, 
                                 self.last_stable_state[i][j],
                                 current_state[i][j]))
        return changes

    def analyze_white_move(self, changes):
        """
        Analyze changes to detect a valid white move
        Returns: (from_square, to_square) or None if no valid move detected
        """
        if not changes or len(changes) != 2:
            return None
            
        # For white's move, we expect:
        # 1. A piece disappearing from ranks 1-2 (board rows 6-7)
        # 2. A piece appearing on a previously empty square
        from_square = None
        to_square = None
        
        for row, col, old_val, new_val in changes:
            if old_val == 1 and new_val == 0 and row >= 6:  # Piece disappeared from white's side
                from_square = (row, col)
            elif old_val == 0 and new_val == 1:  # Piece appeared
                to_square = (row, col)
                
        if from_square and to_square:
            return (from_square, to_square)
            
        return None
    

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
    
    def detect_move_from_frames(self, initial_frame, final_frame):
        """Detect chess move using frame differencing"""
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(final_frame, cv2.COLOR_BGR2GRAY)
        
        # Get absolute difference
        diff = cv2.absdiff(gray1, gray2)
        _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Clean up difference image
        diff = cv2.dilate(diff, None, iterations=4)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        diff = cv2.erode(diff, kernel, iterations=6)
        
        # Find contours of changed regions
        contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area and get the two largest
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(sorted_contours) < 2:
            return None
            
        movement_contours = sorted_contours[:2]
        
        # Get centers of the movement regions
        centers = []
        for contour in movement_contours:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w//2
            center_y = y + int(0.7*h)  # Shifted down to better match piece base
            centers.append((center_x, center_y))
            
        # Convert pixel coordinates to board squares
        squares = []
        for center in centers:
            square = self.pixel_to_square(*center)
            if square:
                squares.append(square)
                
        if len(squares) != 2:
            return None
            
        # Determine which square is source and destination
        if self.game_tracker.is_whites_turn:
            # For white's move, source square should be in lower ranks
            if squares[0][0] > squares[1][0]:  # First square is lower rank
                return (squares[0], squares[1])
            return (squares[1], squares[0])
        else:
            # For black's move, source square should be in upper ranks
            if squares[0][0] < squares[1][0]:  # First square is upper rank
                return (squares[0], squares[1])
            return (squares[1], squares[0])
    
    def pixel_to_square(self, x, y):
        """Convert pixel coordinates to board square coordinates"""
        if not self.calibration_complete:
            return None
            
        h_lines, v_lines = self.get_square_points()
        
        for i in range(8):
            for j in range(8):
                corners = np.array([
                    h_lines[i][j],
                    h_lines[i][j+1],
                    h_lines[i+1][j+1],
                    h_lines[i+1][j]
                ], dtype=np.int32)
                
                if cv2.pointPolygonTest(corners, (x, y), False) >= 0:
                    return (i, j)
        return None
    
    def process_frame(self, frame):
        """Process each video frame"""
        if not self.calibration_complete:
            self.draw_calibration(frame)
            return frame
            
        # Always draw the board grid, regardless of move state
        h_lines, v_lines = self.get_square_points()
        self.draw_board_grid(frame, h_lines, v_lines)
        
        # If we're tracking a move
        if self.move_in_progress:
            # Store initial frame when we first start tracking
            if self.initial_frame is None:
                print("Capturing initial frame for move")
                self.initial_frame = frame.copy()
            
            # Check if movement has stopped
            movement_amount = self.detect_movement(frame)
            cv2.putText(frame, f"Movement: {movement_amount}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if not movement_amount:
                print("Movement stopped, processing move")
                self.final_frame = frame.copy()
                move = self.detect_move_from_frames(self.initial_frame, self.final_frame)
                
                if move:
                    # Update game state
                    if self.game_tracker.update_game_state(move):
                        print(f"Move detected and validated: {move}")
                    else:
                        print("Move was invalid according to chess rules")
                else:
                    print("No valid move detected")
                
                # Reset move tracking
                self.move_in_progress = False
                self.initial_frame = None
                self.final_frame = None
        
        # Draw game state info
        status = "White's turn" if self.game_tracker.is_whites_turn else "Black's turn"
        cv2.putText(frame, status, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        move_status = "Recording move..." if self.move_in_progress else "Press SPACE to start move"
        cv2.putText(frame, move_status, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
        return frame

    def detect_movement(self, frame):
        """Detect if there's significant movement in the frame"""
        if self.last_frame is None:
            self.last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame_diff = cv2.absdiff(gray, self.last_frame)
        
        self.last_frame = gray

        movement = np.sum(frame_diff) > self.movement_threshold
        print(np.sum(frame_diff))
        #cv2.putText(frame, str(movement), (10, 80), 
        #               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if movement:
            self.movement_cooldown = 10  # Wait 10 frames after movement stops
        elif self.movement_cooldown > 0:
            self.movement_cooldown -= 1
            movement = True
            
        return movement
    
    def draw_game_info(self, frame):
        """Draw game state information on frame"""
        status = "White's turn" if self.game_tracker.is_whites_turn else "Black's turn"

    

        cv2.putText(frame, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    
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
    print("\nAfter calibration:")
    print("Press 'space' when you start moving a piece")
    print("Press 'r' to restart calibration")
    print("Press 't' to train the detector")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if chess_vision.is_calibrating:
            chess_vision.draw_calibration(frame)
            cv2.imshow('Chess Vision', frame)
        else:
            processed_frame = chess_vision.process_frame(frame)
            
            # Add game state info to frame
            status = "White's turn" if chess_vision.game_tracker.is_whites_turn else "Black's turn"
            move_status = "Recording move..." if chess_vision.move_in_progress else "Waiting for move..."
            cv2.putText(processed_frame, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(processed_frame, move_status, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Chess Vision', processed_frame)
        
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
        elif key == ord(' '):  # Spacebar to start move recording
            if chess_vision.game_tracker.waiting_for_move:
                chess_vision.move_in_progress = True
                chess_vision.initial_frame = None  # Reset to ensure we get a fresh frame
                print("Started recording move...")
            else:
                print("Not waiting for move - it might be the opponent's turn")

        # Collect training data every N frames after calibration
        if chess_vision.is_collecting_data and not chess_vision.piece_detector.is_trained:
            frame_count += 1
            if frame_count >= 30:
                frame_count = 0
    
    
   # cap.release()
  #  cv2.destroyAllWindows()

if __name__ == "__main__":
    test_webcam()