import cv2
import numpy as np

class SimpleChessDetector:
    def __init__(self):
        self.points = []  # Calibration points
        self.is_calibrated = False
        self.first_frame = None
        self.second_frame = None
        
    def add_calibration_point(self, x, y):
        """Add a calibration point and check if calibration is complete"""
        if len(self.points) < 4:
            self.points.append((x, y))
            print(f"Added point {len(self.points)}: ({x}, {y})")
            
        if len(self.points) == 4:
            self.is_calibrated = True
            print("Calibration complete!")
            return True
        return False
    
    def get_board_squares(self):
        """Calculate grid points with perspective correction"""
        if not self.is_calibrated:
            return None
            
        pts = np.float32(self.points)
        h_lines = []
        v_lines = []
        
        # Calculate horizontal and vertical grid lines
        for i in range(9):
            # Horizontal lines
            left = pts[0] + (pts[3] - pts[0]) * i / 8.0
            right = pts[1] + (pts[2] - pts[1]) * i / 8.0
            h_line = [left + (right - left) * j / 8.0 for j in range(9)]
            h_lines.append(h_line)
            
            # Vertical lines
            top = pts[0] + (pts[1] - pts[0]) * i / 8.0
            bottom = pts[3] + (pts[2] - pts[3]) * i / 8.0
            v_line = [top + (bottom - top) * j / 8.0 for j in range(9)]
            v_lines.append(v_line)
            
        return h_lines, v_lines
    
    def detect_move(self):
        """Detect chess move by comparing frames using improved image processing"""
        if self.first_frame is None or self.second_frame is None:
            print("Missing frames for comparison")
            return None
            
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.second_frame, cv2.COLOR_BGR2GRAY)
        
        # Get absolute difference with improved thresholding
        diff = cv2.absdiff(gray1, gray2)
        print(f"Max difference value: {np.max(diff)}")  # Debug print
        
        _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Enhanced noise removal technique
        diff = cv2.dilate(diff, None, iterations=4)
        kernel_size = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        diff = cv2.erode(diff, kernel, iterations=6)
        
        # Store the debug image before finding contours
        self.debug_image = diff.copy()
        
        # Find contours and sort by area
        contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Found {len(contours)} contours")  # Debug print
        
        if not contours:
            print("No contours found")
            return None
        
        # Sort contours by area
        sorted_contours = sorted([(c, cv2.contourArea(c)) for c in contours], 
                            key=lambda x: x[1], reverse=True)
        
        # Print areas of largest contours
        for i, (_, area) in enumerate(sorted_contours[:3]):
            print(f"Contour {i} area: {area}")
        
        # Get the two largest contours
        if len(sorted_contours) < 2:
            print("Not enough large contours found")
            return None
        
        # Get bounding boxes and calculate centers
        centers = []
        for contour, area in sorted_contours[:2]:
            if area < 100:  # Minimum area threshold
                print(f"Contour too small: {area}")
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w//2
            center_y = y + int(0.7 * h)
            centers.append((center_x, center_y))
            
            # Draw contours on debug image (in color)
            cv2.drawContours(self.debug_image, [contour], -1, 255, 2)
            cv2.circle(self.debug_image, (center_x, center_y), 5, 255, -1)
        
        print(f"Found {len(centers)} centers: {centers}")  # Debug print
        
        # Convert pixel coordinates to board squares
        squares = []
        for center in centers:
            square = self.pixel_to_square(*center)
            if square:
                squares.append(square)
                print(f"Center {center} maps to square {square}")  # Debug print
        
        if len(squares) != 2:
            print(f"Invalid number of squares detected: {len(squares)}")
            return None
        
        # Convert squares to algebraic notation
        moves = [self.square_to_algebraic(*sq) for sq in squares]
        print(f"Detected move: {moves[0]}{moves[1]}")  # Debug print
        
        return f"{moves[0]}{moves[1]}"

    def draw_debug(self, frame):
        """Draw debug information on the frame"""
        if hasattr(self, 'debug_image') and self.debug_image is not None:
            # Draw the difference image in a corner
            h, w = self.debug_image.shape
            scale = 0.3
            small_debug = cv2.resize(self.debug_image, 
                                (int(w * scale), int(h * scale)))
            
            # Convert to BGR for overlay
            small_debug_colored = cv2.cvtColor(small_debug, cv2.COLOR_GRAY2BGR)
            
            # Overlay in top-right corner
            frame_h, frame_w = frame.shape[:2]
            x_offset = frame_w - small_debug_colored.shape[1] - 10
            y_offset = 10
            
            try:
                # Create region of interest
                roi = frame[y_offset:y_offset + small_debug_colored.shape[0], 
                        x_offset:x_offset + small_debug_colored.shape[1]]
                
                # Check if ROI has the same shape as small_debug_colored
                if roi.shape == small_debug_colored.shape:
                    frame[y_offset:y_offset + small_debug_colored.shape[0], 
                        x_offset:x_offset + small_debug_colored.shape[1]] = small_debug_colored
                else:
                    print("ROI shape mismatch:", roi.shape, small_debug_colored.shape)
            except Exception as e:
                print("Error in draw_debug:", str(e))
    
    def pixel_to_square(self, x, y):
        """Convert pixel coordinates to board square coordinates"""
        if not self.is_calibrated:
            return None
            
        h_lines, v_lines = self.get_board_squares()
        
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
    
    def square_to_algebraic(self, row, col):
        """Convert row,col coordinates to algebraic notation"""
        file = chr(ord('a') + col)
        rank = str(8 - row)
        return file + rank
    
    def draw_board(self, frame):
        """Draw the chess board grid and debug information on the frame"""
        if not self.is_calibrated:
            # Draw calibration points
            for i, point in enumerate(self.points):
                cv2.circle(frame, point, 5, (0, 0, 255), -1)
                cv2.putText(frame, str(i+1), (point[0]+10, point[1]+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Draw grid lines
            h_lines, v_lines = self.get_board_squares()
            
            # Draw horizontal lines
            for i in range(9):
                points = np.array([h_lines[i][j] for j in range(9)], dtype=np.int32)
                cv2.polylines(frame, [points.reshape(-1, 1, 2)], False, (0, 255, 0), 2)
                
            # Draw vertical lines
            for j in range(9):
                points = np.array([v_lines[i][j] for i in range(9)], dtype=np.int32)
                cv2.polylines(frame, [points.reshape(-1, 1, 2)], False, (0, 255, 0), 2)
            
            # If we have detected centers, draw them
            if hasattr(self, 'last_centers') and self.last_centers:
                for center in self.last_centers:
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    # Draw coordinate text
                    cv2.putText(frame, f"({center[0]}, {center[1]})", 
                            (center[0]+10, center[1]+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    # Try to map to square and display
                    square = self.pixel_to_square(*center)
                    if square:
                        algebraic = self.square_to_algebraic(*square)
                        cv2.putText(frame, algebraic, 
                                (center[0]-10, center[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

def mouse_callback(event, x, y, flags, param):
    detector = param
    if event == cv2.EVENT_LBUTTONDOWN and not detector.is_calibrated:
        detector.add_calibration_point(x, y)

def main():
    cap = cv2.VideoCapture(0)
    detector = SimpleChessDetector()
    
    cv2.namedWindow('Chess Detector')
    cv2.setMouseCallback('Chess Detector', mouse_callback, detector)
    
    print("Simple Chess Move Detector")
    print("1. Click the four corners of the board (top-left, top-right, bottom-right, bottom-left)")
    print("2. Press 'f' to capture first frame before moving piece")
    print("3. Move your piece")
    print("4. Press 's' to capture second frame")
    print("5. Press 'd' to detect move")
    print("Press 'q' to quit, 'r' to reset calibration")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw the board grid or calibration points
        detector.draw_board(frame)
        
        if detector.first_frame is not None and detector.second_frame is not None:
            detector.draw_debug(frame)

        # Add status text
        if not detector.is_calibrated:
            status = f"Calibrating: {len(detector.points)}/4 points"
        elif detector.first_frame is None:
            status = "Press 'f' to capture first frame"
        elif detector.second_frame is None:
            status = "Press 's' to capture second frame"
        elif key == ord('d') and detector.first_frame is not None and detector.second_frame is not None:
            move = detector.detect_move()
            if move:
                print(f"Detected move: {move}")
            else:
                print("No move detected")
            
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Chess Detector', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.points = []
            detector.is_calibrated = False
            detector.first_frame = None
            detector.second_frame = None
            print("Calibration reset")
        elif key == ord('f') and detector.is_calibrated:
            detector.first_frame = frame.copy()
            print("Captured first frame")
        elif key == ord('s') and detector.first_frame is not None:
            detector.second_frame = frame.copy()
            print("Captured second frame")
        elif key == ord('d') and detector.first_frame is not None and detector.second_frame is not None:
            move = detector.detect_move()
            detector.draw_debug(frame)
            if move:
                print(f"Detected move: {move}")
            else:
                print("No move detected")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()