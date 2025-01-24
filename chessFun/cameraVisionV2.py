import cv2
import numpy as np
import time

class ChessVision:
    def __init__(self):
        self.points = [(118, 78), (464, 62), (487, 416), (126, 431)]
        self.last_board_state = self.get_initial_board_state()
        self.h_lines = None
        self.v_lines = None
        self.last_frame = None
        self.movement_cooldown = 0
        self.movement_threshold = 2400000  # Adjust this value based on sensitivity needed
        
    def get_initial_board_state(self):
        """Initialize 8x8 board with starting position (1 for piece, 0 for empty)"""
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
        print(np.sum(frame_diff))
        if movement:
            self.movement_cooldown = 10  # Wait 10 frames after movement stops
        elif self.movement_cooldown > 0:
            self.movement_cooldown -= 1
            movement = True
            
        return movement
    
    def analyze_square(self, frame, square_corners):
        """Determine if a square is empty (pure black/white) based on color variance"""
        # Convert corners to integer points
        corners = [tuple(map(int, corner)) for corner in square_corners]
        
        # Create a mask for the square
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(corners)], 255)
        
        # Extract the square region
        roi = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Convert to HSV for better color analysis
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate statistics for the square
        mean_val = cv2.mean(hsv_roi, mask=mask)
        std_val = cv2.meanStdDev(hsv_roi, mask=mask)[1]
        
        # Check if the square is purely black or white (low saturation and low standard deviation)
        is_empty = (
            std_val[0][0] < 15 and  # Low variation in hue
            std_val[1][0] < 20 and  # Low variation in saturation
            (mean_val[2] < 50 or mean_val[2] > 200)  # Very dark or very bright
        )
        
        return is_empty
    
    def get_move_notation(self, from_square, to_square):
        """Convert square coordinates to chess notation"""
        files = 'abcdefgh'
        from_notation = f"{files[from_square[1]]}{8-from_square[0]}"
        to_notation = f"{files[to_square[1]]}{8-to_square[0]}"
        return f"{from_notation}{to_notation}"
    
    def process_frame(self, frame):
        # Check for movement
        if self.detect_movement(frame):
            # If movement detected, just draw the grid without analysis
            h_lines, v_lines = self.get_square_points()
            for i in range(8):
                for j in range(8):
                    corners = [
                        h_lines[i][j],
                        h_lines[i][j+1],
                        h_lines[i+1][j+1],
                        h_lines[i+1][j]
                    ]
                    for k in range(4):
                        pt1 = tuple(map(int, corners[k]))
                        pt2 = tuple(map(int, corners[(k+1)%4]))
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
            
            cv2.putText(frame, "Movement detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
            
        h_lines, v_lines = self.get_square_points()
        current_board_state = np.zeros((8, 8), dtype=int)
        
        # Analyze each square
        for i in range(8):
            for j in range(8):
                corners = [
                    h_lines[i][j],
                    h_lines[i][j+1],
                    h_lines[i+1][j+1],
                    h_lines[i+1][j]
                ]
                
                # Check if square is empty (pure black/white)
                is_empty = self.analyze_square(frame, corners)
                current_board_state[i][j] = 0 if is_empty else 1
                
                # Draw grid
                for k in range(4):
                    pt1 = tuple(map(int, corners[k]))
                    pt2 = tuple(map(int, corners[(k+1)%4]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
                
                # Only draw green circle on empty squares
                if is_empty:
                    center = tuple(map(int, sum(corners)/4))
                    cv2.circle(frame, center, 3, (0, 255, 0), -1)
        
        # Detect moves by comparing with previous state
        if not np.array_equal(current_board_state, self.last_board_state):
            diff = current_board_state - self.last_board_state
            source = np.where(diff == -1)
            dest = np.where(diff == 1)
            
            if len(source[0]) == 1 and len(dest[0]) == 1:
                move = self.get_move_notation(
                    (source[0][0], source[1][0]),
                    (dest[0][0], dest[1][0])
                )
                print(f"Move detected: {move}")
        
        self.last_board_state = current_board_state
        return frame

def test_webcam():
    cap = cv2.VideoCapture(0)
    chess_vision = ChessVision()
    
    print("Chess Vision started. Make moves on the board.")
    print("Press 'q' to quit")
    print("Press 'r' to reset board state")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame = chess_vision.process_frame(frame)
        cv2.imshow('Chess Vision', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            chess_vision.last_board_state = chess_vision.get_initial_board_state()
            print("Board state reset to initial position")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_webcam()