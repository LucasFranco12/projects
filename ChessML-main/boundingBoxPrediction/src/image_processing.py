import cv2
import numpy as np

def warp_image(image: np.ndarray, corners: dict, target_size: tuple = (800, 800)) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Warps an image based on corner coordinates.

    Args:
        image: The input image (NumPy array).
        corners: A dictionary with keys 'top_left', 'top_right', 
                 'bottom_right', 'bottom_left'. Each value should be a list 
                 or tuple of [x, y] coordinates.
        target_size: The desired output size (width, height).

    Returns:
        A tuple containing:
        - warped_image: The perspective-warped image (NumPy array) or None if corners are invalid.
        - matrix: The perspective transformation matrix or None if corners are invalid.
    """
    try:
        src_pts = np.array([
            corners['top_left'],
            corners['top_right'],
            corners['bottom_right'],
            corners['bottom_left']
        ], dtype=np.float32)

        dst_pts = np.array([
            [0, 0],
            [target_size[0]-1, 0],
            [target_size[0]-1, target_size[1]-1],
            [0, target_size[1]-1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, matrix, target_size)
        return warped, matrix
    except KeyError as e:
        print(f"Error: Missing corner key in input: {e}")
        return None, None
    except Exception as e:
        print(f"Error during image warping: {e}")
        return None, None

def get_corners_manual(image: np.ndarray) -> dict | None:
    """Allows user to manually click 4 corners on the displayed image.
    
    Returns corner dict or None if window closed.
    """
    # Define a maximum display size
    MAX_DISPLAY_WIDTH = 800
    MAX_DISPLAY_HEIGHT = 600

    # Resize image for display while keeping aspect ratio
    def resize_for_display(img):
        h, w = img.shape[:2]
        scale = min(MAX_DISPLAY_WIDTH / w, MAX_DISPLAY_HEIGHT / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale

    # Resize image for display
    image_display, scale = resize_for_display(image.copy())
    original_shape = image.shape[:2]
    display_shape = image_display.shape[:2]
    print(f"Original image size: {original_shape[1]}x{original_shape[0]}")
    print(f"Display size: {display_shape[1]}x{display_shape[0]}, scale: {scale:.3f}")

    corners = []
    window_name = "Click Corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left (Press Q to Quit)"

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(corners) < 4:
                # Convert display coordinates back to original image coordinates
                orig_x, orig_y = int(x / scale), int(y / scale)
                corners.append([orig_x, orig_y])
                
                # Draw feedback on the display image
                cv2.circle(image_display, (x, y), 5, (0, 255, 0), -1) # Green circle
                cv2.putText(image_display, f"{len(corners)}", (x+10, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(window_name, image_display)
                
                print(f"Corner {len(corners)} selected at display position ({x},{y}), original position ({orig_x},{orig_y})")
                
                if len(corners) == 4:
                    print("4 corners selected. Press any key (except Q) to confirm, or Q to quit.")
    
    # Create window first with AUTOSIZE to fit the resized image
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, click_event)
    
    print("Please click the 4 corners in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left")
    print("Press 'Q' to quit without selecting.")

    while True:
        cv2.imshow(window_name, image_display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit signal received.")
            cv2.destroyAllWindows()
            return None # User quit
        if len(corners) == 4 and key != 255: # Any key other than 'q' (and not no key) confirms
            print("Corners confirmed.")
            break
            
    cv2.destroyAllWindows()

    if len(corners) == 4:
        return {
            'top_left': corners[0],
            'top_right': corners[1],
            'bottom_right': corners[2],
            'bottom_left': corners[3]
        }
    else:
        return None # Should not happen if loop broken correctly, but safety check

# Placeholder for future automatic detection
# def find_chessboard_corners_auto(image: np.ndarray):
#     # Implementation using cv2.findChessboardCorners etc.
#     pass 