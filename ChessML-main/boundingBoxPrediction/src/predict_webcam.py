import cv2
import sys
from pathlib import Path
import numpy as np
import time

script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent)) 

from src.models.yolo_model import ChessPieceDetector
from src.utils.image_processing import warp_image, get_corners_manual


WEIGHTS_PATH = r'c:\Users\lucas\Downloads\PROGRAMS\runs\detect\train11\weights\best.pt'

# Prediction settings
CONFIDENCE_THRESHOLD = 0.3
PREDICTION_INTERVAL = 25  # Run prediction every N frames

# Board dimensions
TARGET_SIZE = (800, 800)  # Size for processing/prediction
DISPLAY_SIZE = (1000, 1000)  # Size for displaying windows

# Board visualization
SHOW_GRID = True  # Show chess grid
CAMERA_INDEX = 0  # Default webcam

def create_board_state(predictions, target_size=TARGET_SIZE):
    """Create a chess board state from predictions."""
    board = [["." for _ in range(8)] for _ in range(8)]
    square_size = target_size[0] // 8
    
    for pred in predictions:
        x1, y1, x2, y2 = pred['bbox']
        # Calculate center point
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Convert to board coordinates (0-7)
        col = int(cx / square_size)
        row = int(cy / square_size)
        
        if 0 <= row < 8 and 0 <= col < 8:
            board[row][col] = (pred['class_name'], pred['confidence'])
    
    return board

def print_board_state(board):
    """Print the current board state in a readable format."""
    print("\nBoard state:")
    print("  a b c d e f g h")
    for i, row in enumerate(board):
        row_str = []
        for square in row:
            if square == ".":
                row_str.append(".")
            else:
                piece_name, _ = square
                # Convert piece name to letter
                if "white" in piece_name:
                    prefix = "w"
                else:
                    prefix = "b"
                piece_type = piece_name.split("_")[1][0].upper()
                row_str.append(f"{prefix}{piece_type}")
        print(f"{i+1} {' '.join(row_str)} {i+1}")
    print("  a b c d e f g h")

def main():
    # --- Initialize Detector --- 
    try:
        detector = ChessPieceDetector(weights_path=WEIGHTS_PATH)
        print(f"Model loaded from: {WEIGHTS_PATH}")
    except Exception as e:
        print(f"Error initializing detector: {e}")
        sys.exit(1)

    # --- Initialize Webcam --- 
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open webcam")
        sys.exit(1)
    
    print("Webcam opened successfully")
    
    # --- Calibration Phase --- 
    corners = None
    matrix = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame for calibration.")
            cap.release()
            sys.exit(1)

        display_frame = frame.copy()
        cv2.putText(display_frame, "Press 'C' to Calibrate (click corners), 'Q' to Quit", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Webcam - Calibration", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quitting.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        elif key == ord('c'):
            print("Starting corner selection...")
            cv2.destroyWindow("Webcam - Calibration") # Close calibration prompt
            corners = get_corners_manual(frame) # Use the current frame for selection
            if corners:
                # Try warping immediately to get the matrix
                _ , test_matrix = warp_image(frame, corners, TARGET_SIZE)
                if test_matrix is not None:
                    matrix = test_matrix
                    print("Calibration successful. Starting prediction.")
                    break # Exit calibration loop
                else:
                    print("Warping failed with selected corners. Please try calibration again.")
            else:
                print("Corner selection cancelled. Press 'C' again or 'Q' to quit.")

    # --- Create inverse matrix for overlay mode ---
    # Create source and destination points for inverse transformation
    src_pts = np.array([
        [0, 0],
        [TARGET_SIZE[0]-1, 0],
        [TARGET_SIZE[0]-1, TARGET_SIZE[1]-1],
        [0, TARGET_SIZE[1]-1]
    ], dtype=np.float32)
    
    dst_pts = np.array([
        corners['top_left'],
        corners['top_right'],
        corners['bottom_right'],
        corners['bottom_left']
    ], dtype=np.float32)
    
    # Create inverse matrix to warp back to original image space
    inv_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Create a mask for blending
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    board_contour = np.array([corners['top_left'], corners['top_right'], 
                            corners['bottom_right'], corners['bottom_left']], dtype=np.int32)
    cv2.fillPoly(mask, [board_contour], 255)

    # --- Prediction Loop --- 
    fps = 0
    frame_count = 0
    prediction_frame_counter = 0
    start_time = time.time()
    
    # Initialize variables to store the last valid predictions and board state
    last_predictions = []
    last_board_state = [["." for _ in range(8)] for _ in range(8)]

    print(f"\nRunning predictions every {PREDICTION_INTERVAL} frames")
    print("Press 'q' to quit, 'r' to recalibrate")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Warp the current frame using the calibration matrix
        warped_board = cv2.warpPerspective(frame, matrix, TARGET_SIZE)
        
        prediction_frame_counter += 1
        
        # Only run prediction every PREDICTION_INTERVAL frames
        if prediction_frame_counter >= PREDICTION_INTERVAL:
            prediction_frame_counter = 0
            
            # Perform prediction on the processed board
            try:
                results = detector.predict(warped_board, confidence_threshold=CONFIDENCE_THRESHOLD)
                new_predictions = detector.get_predictions_details(results)
                
                # Only update predictions if we got valid results
                if len(new_predictions) > 0:
                    last_predictions = new_predictions
                    last_board_state = create_board_state(last_predictions)
                    
                    # Print board state to console
                    print_board_state(last_board_state)
                    print(f"Updated predictions: {len(last_predictions)} pieces detected")
                
            except Exception as e:
                print(f"Prediction error: {e}")
                # Continue with previous predictions if error occurs
        
        # Create display image from original frame
        display_img = frame.copy()
        
        # Create warped view with highlighted squares
        warped_with_predictions = warped_board.copy()
        h, w = warped_with_predictions.shape[:2]
        square_size = w // 8
        
        # Draw the chess board grid
        for i in range(1, 8):
            cv2.line(warped_with_predictions, (i * square_size, 0), 
                     (i * square_size, h), (128, 128, 128), 1)
            cv2.line(warped_with_predictions, (0, i * square_size), 
                     (w, i * square_size), (128, 128, 128), 1)
            
        # Add board coordinates if requested
        if SHOW_GRID:
            for i in range(8):
                # Column labels (a-h)
                cv2.putText(warped_with_predictions, chr(97 + i), 
                           (i * square_size + square_size//2 - 5, h - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                # Row labels (1-8) 
                cv2.putText(warped_with_predictions, str(i + 1), 
                           (5, i * square_size + square_size//2 + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Highlight squares based on saved board state
        for row in range(8):
            for col in range(8):
                square = last_board_state[row][col]
                if square != ".":
                    piece_name, conf = square
                    
                    # Define square coordinates
                    x1 = col * square_size
                    y1 = row * square_size
                    x2 = (col + 1) * square_size
                    y2 = (row + 1) * square_size
                    
                    # Color code based on piece type
                    if "white" in piece_name:
                        color = (0, 255, 0)  # Green for white pieces
                        text_color = (0, 0, 0)  # Black text
                    else:
                        color = (0, 0, 255)  # Red for black pieces
                        text_color = (255, 255, 255)  # White text
                        
                    # Create semi-transparent overlay
                    overlay = warped_with_predictions.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    
                    # Apply transparency
                    alpha = 0.3  # Transparency factor
                    cv2.addWeighted(overlay, alpha, warped_with_predictions, 1 - alpha, 0, warped_with_predictions)
                    
                    # Add piece name label
                    piece_label = piece_name.split('_')[1]  # Get just 'pawn', 'knight', etc.
                    cv2.putText(warped_with_predictions, piece_label, 
                               (x1 + 5, y1 + square_size//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                    
                    # Add confidence as percentage
                    conf_str = f"{conf:.0%}"
                    cv2.putText(warped_with_predictions, conf_str, 
                               (x1 + 5, y1 + square_size//2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        
        # Draw bounding boxes directly from predictions
        for pred in last_predictions:
            # Get bounding box coordinates
            x_min, y_min, x_max, y_max = map(int, pred['bbox'])
            piece_name = pred['class_name']
            conf = pred['confidence']
            
            # Set colors - magenta for white pieces, cyan for black
            if "white" in piece_name:
                box_color = (255, 0, 255)  # Magenta
            else:
                box_color = (255, 255, 0)  # Cyan
                
            # Draw rectangle with thicker lines
            cv2.rectangle(warped_with_predictions, (x_min, y_min), (x_max, y_max), box_color, 2)
            
            # Add small confidence indicator at top-right of box
            mini_conf = f"{conf:.2f}"
            cv2.putText(warped_with_predictions, mini_conf, 
                       (x_max - 50, y_min - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
        
        # Warp predictions back to original frame
        warped_back = cv2.warpPerspective(warped_with_predictions, inv_matrix, 
                                         (frame.shape[1], frame.shape[0]))
        
        # Combine original frame with warped predictions using mask
        display_img = cv2.bitwise_and(display_img, display_img, mask=cv2.bitwise_not(mask))
        display_img = cv2.add(display_img, cv2.bitwise_and(warped_back, warped_back, mask=mask))
        
        # Display FPS on the final image
        cv2.putText(display_img, f"FPS: {fps:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the result - resize for better visibility
        display_resized = cv2.resize(display_img, DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imshow("Chess Piece Prediction (Overlay)", display_resized)
        
        # Also show the warped view in a separate window
        cv2.namedWindow("Warped Board View", cv2.WINDOW_NORMAL)
        warped_resized = cv2.resize(warped_with_predictions, DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imshow("Warped Board View", warped_resized)
        cv2.resizeWindow("Warped Board View", DISPLAY_SIZE[0], DISPLAY_SIZE[1])

        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
            
            # Display current frame count and prediction interval
            detected_pieces = sum(1 for row in last_board_state for square in row if square != ".")
            print(f"FPS: {fps:.2f}, Detected pieces: {detected_pieces}, Next prediction in {PREDICTION_INTERVAL - prediction_frame_counter} frames")

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting prediction loop.")
            break
        # Press 'r' to recalibrate
        elif cv2.waitKey(1) & 0xFF == ord('r'):
            print("Recalibrating...")
            cv2.destroyAllWindows()
            corners = get_corners_manual(frame)
            if corners:
                _, matrix = warp_image(frame, corners, TARGET_SIZE)
                # Update inverse matrix and mask for overlay mode
                src_pts = np.array([
                    [0, 0],
                    [TARGET_SIZE[0]-1, 0],
                    [TARGET_SIZE[0]-1, TARGET_SIZE[1]-1],
                    [0, TARGET_SIZE[1]-1]
                ], dtype=np.float32)
                
                dst_pts = np.array([
                    corners['top_left'],
                    corners['top_right'],
                    corners['bottom_right'],
                    corners['bottom_left']
                ], dtype=np.float32)
                
                inv_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                
                mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                board_contour = np.array([corners['top_left'], corners['top_right'], 
                                        corners['bottom_right'], corners['bottom_left']], dtype=np.int32)
                cv2.fillPoly(mask, [board_contour], 255)
                print("Recalibration complete.")

    # --- Cleanup --- 
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and windows closed.")

if __name__ == "__main__":
    main() 