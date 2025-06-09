# Chess Arena Controller
# This class handles the calibration and control of the chess arena board. It allows the user to:
# - Calibrate the board by selecting the four corners of the arena.
# - Calculate the size of the board and individual squares.
# - Execute moves on the arena board by simulating mouse clicks.

import pyautogui
import numpy as np
import cv2
import time

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

    #  Calculate the actual board size and square size from the calibration points.
    # This method uses the four corner points of the arena to calculate: The width and height of the board and the size of each square on the board
    def calculate_board_size(self):
        
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

    # Get the center of a square on the chess board in screen coordinates.
    # This method takes a square (row, col) as input and returns the screen coordinates of the center of that square.
    def get_square_center(self, square):

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

    # Execute a move on the chess arena board by simulating mouse clicks.
    # This method takes the source and destination squares as input and simulates the mouse movements and clicks to perform the move.
    def execute_move(self, from_square, to_square):
        
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

    # Add a corner point to the arena calibration.
    # This method takes the x and y coordinates of the corner point as input and adds it to the list of arena corners.
    def add_corner(self, x=None, y=None):
        if x is None or y is None:
            x, y = pyautogui.position()
            
        point_names = ["top-left", "top-right", "bottom-right", "bottom-left"]
        current_point = len(self.arena_corners)
        
        self.arena_corners.append((x, y))
        print(f"Arena calibration: Added {point_names[current_point]} point at ({x}, {y})")
        
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

    # Start the calibration process for the chess arena board.
    def start_calibration(self):
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