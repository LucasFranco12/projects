import cv2
import numpy as np
from typing import Dict, Tuple, Optional

class ChessMoveDetector:
    def __init__(self):
        self.squares = {}  # Coordinates of all squares
        self.initial_frame = None  # Frame before the move
        self.moved_frame = None  # Frame after the move
        self.square_size = None

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Basic preprocessing to improve board detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    def find_board(self, thresh: np.ndarray) -> Optional[np.ndarray]:
        """Detect the chessboard in the image."""
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        board_contour = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 300:  # Ignore small contours
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and area > max_area:
                max_area = area
                board_contour = approx

        return board_contour

    def get_square_coordinates(self, board_corners: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate coordinates for all squares."""
        pts = board_corners.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left

        width = np.linalg.norm(rect[1] - rect[0])
        height = np.linalg.norm(rect[3] - rect[0])
        self.square_size = min(width, height) / 8

        squares = {}
        for row in range(8):
            for col in range(8):
                top_left = rect[0] + col * (rect[1] - rect[0]) / 8 + row * (rect[3] - rect[0]) / 8
                top_right = top_left + (rect[1] - rect[0]) / 8
                bottom_right = top_right + (rect[3] - rect[0]) / 8
                bottom_left = top_left + (rect[3] - rect[0]) / 8
                square_name = f"{chr(col + 97)}{8-row}"
                squares[square_name] = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)

        return squares

    def detect_changes(self, initial_frame: np.ndarray, moved_frame: np.ndarray) -> Tuple[Optional[str], Optional[str]]:
        """Detect the square where the piece moved from and to."""
        initial_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
        moved_gray = cv2.cvtColor(moved_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(initial_gray, moved_gray)
        _, diff_thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

        moved_from = None
        moved_to = None

        for square_name, corners in self.squares.items():
            mask = np.zeros(diff_thresh.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [corners], 255)
            masked_diff = cv2.bitwise_and(diff_thresh, diff_thresh, mask=mask)
            change = cv2.countNonZero(masked_diff)

            if change > 300:  # Adjust threshold based on setup
                if moved_from is None:
                    moved_from = square_name
                elif moved_to is None:
                    moved_to = square_name

        return moved_from, moved_to

    def draw_squares(self, frame: np.ndarray, highlight: Tuple[Optional[str], Optional[str]] = (None, None)) -> np.ndarray:
        """Draw the board with optional highlights."""
        annotated_frame = frame.copy()
        for square_name, corners in self.squares.items():
            color = (0, 255, 0)
            if square_name in highlight:
                color = (0, 0, 255)
            cv2.polylines(annotated_frame, [corners], True, color, 2)
        return annotated_frame

def main():
    cap = cv2.VideoCapture(0)
    detector = ChessMoveDetector()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('d'):  # Detect board
            processed = detector.preprocess_frame(frame)
            board_corners = detector.find_board(processed)
            if board_corners is not None:
                detector.squares = detector.get_square_coordinates(board_corners)
                detector.initial_frame = frame.copy()
                print("Board detected. Initial frame captured.")
            else:
                print("Board not detected. Please adjust the camera.")

        elif key == ord('m'):  # Capture moved frame and detect changes
            if detector.initial_frame is not None:
                detector.moved_frame = frame.copy()
                moved_from, moved_to = detector.detect_changes(detector.initial_frame, detector.moved_frame)
                if moved_from and moved_to:
                    print(f"Piece moved from {moved_from} to {moved_to}")
                    annotated_frame = detector.draw_squares(frame, highlight=(moved_from, moved_to))
                    cv2.imshow('Chess Move Detection', annotated_frame)
                else:
                    print("No movement detected.")
            else:
                print("No initial frame captured. Press 'd' to detect the board first.")

        elif key == ord('q'):  # Quit the program
            break

        if detector.squares:
            annotated_frame = detector.draw_squares(frame)
            cv2.imshow('Chess Move Detection', annotated_frame)
        else:
            cv2.imshow('Chess Move Detection', frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
