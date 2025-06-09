#!/usr/bin/env python3
# Chess Piece Detector Main
# This script allows you to play chess against a robot arm.
# It uses computer vision to detect chess pieces and moves, and controls a robot arm to execute moves.
# Execution commands:
# - Ensure webcam permissions: ls -l /dev/video*
# - Grant permissions: sudo chmod 666 /dev/video2 (adjust for your device)
# - Build the workspace: catkin_make
# - Source the setup file: source devel/setup.bash
# - Grant microcontroller permissions: sudo chmod 666 /dev/ttyUSB0
# - Launch the robot controller: roslaunch open_manipulator_controller open_manipulator_controller.launch use_platform:=true
# - Launch the robot arm: roslaunch open_manipulator_controller_gui open_manipulator_controller_gui
# - Launch this script: roslaunch chessML ChessPieceDetectorMain.py

# Import necessary libraries
import cv2
import numpy as np
import time
import os
import rospy
from std_msgs.msg import String
import pyautogui

# Import the necessary classes
from chess_piece_detector import ChessPieceDetector
from robot_chess_controller import RobotChessController, json_trajectory
from chess_arena_controller import ChessArenaController


def main():
    # Initialize the ROS node
    rospy.init_node("chess_piece_detector", anonymous=True)
    pub = rospy.Publisher("/chess_moves", String, queue_size=10)
    cap = cv2.VideoCapture(2, cv2.CAP_V4L2)  # Open the webcam (adjust index as needed)

    # Set the resolution to 1080p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize the chess piece detector and robot controller
    detector = ChessPieceDetector()
    robot_controller = RobotChessController()
    frame_count = 0

    # Print instructions for the user
    print("Chess Piece Detector started.")
    print("Please calibrate the board first:")
    print("1. Click the top-left corner")
    print("2. Click the top-right corner")
    print("3. Click the bottom-right corner")
    print("4. Click the bottom-left corner")
    print("Press '1' to capture frame with no pieces.")
    print("Press '2' to capture frame with white pieces on lower half.")
    print("Press '3' to capture frame with black pieces on upper half.")
    print("Press '4' to capture frame with normal chess start.")
    print("Press 't' to train the model.")
    print("Press 's' to save the data.")
    print("Press 'l' to load the data.")
    print("Press 'q' to quit.")
    print("Additional instructions:")
    print("Press 'a' to start Arena board calibration")
    print(
        "When in Arena calibration mode, click the four corners of the Arena board in order:"
    )
    print("1. Top-left")
    print("2. Top-right")
    print("3. Bottom-right")
    print("4. Bottom-left")

    # Mouse callback function for handling clicks and dragging
    def mouse_callback(event, x, y, flags, param):
        detector = param
        if event == cv2.EVENT_LBUTTONDOWN:
            # Handle Arena calibration mode
            if detector.arena_calibration_mode:
                if detector.arena_controller.add_corner(x, y):
                    detector.arena_calibration_mode = False
                    print("Arena calibration complete!")
                return  # Prevent regular calibration handling
            # Handle board calibration
            elif detector.is_calibrating:
                if detector.add_calibration_point(x, y):
                    print("Calibration complete! Now collecting training data...")
                    print("Press 'T' when ready to train the detector")
            else:
                # Handle selection of corners for adjustment
                print(f"Clicked at ({x}, y)")
                for i, points in enumerate(detector.corners):
                    for j, p in enumerate(points):
                        if np.linalg.norm(np.array(p) - np.array([x, y])) < 10:
                            detector.selected_point = (i, j)
                            print(f"Selected point: {detector.selected_point}")
                            break
        elif event == cv2.EVENT_MOUSEMOVE and detector.selected_point is not None:
            # Update the position of the selected corner
            i, j = detector.selected_point
            detector.update_shared_corners(i, j, (x, y))
            print(f"Moved point {detector.selected_point} to ({x}, {y})")
        elif event == cv2.EVENT_LBUTTONUP:
            # Finalize the position of the selected corner
            if detector.selected_point is not None:
                i, j = detector.selected_point
                detector.update_shared_corners(i, j, (x, y))
                detector.selected_point = None

    # Set up the OpenCV window and mouse callback
    cv2.namedWindow("Chess Piece Detector")
    cv2.setMouseCallback("Chess Piece Detector", mouse_callback, detector)

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam. Exiting...")
            break

        display_frame = frame.copy()

        # Handle Arena calibration mode
        if detector.arena_controller.calibrating:
            # Show instructions on the frame
            cv2.putText(
                display_frame,
                "Arena Calibration Mode - Move mouse to corners and press SPACE",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

            # Draw current mouse position
            x, y = pyautogui.position()
            cv2.putText(
                display_frame,
                f"Mouse: ({x}, {y})",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

            # Draw already placed points
            for i, point in enumerate(detector.arena_controller.arena_corners):
                cv2.putText(
                    display_frame,
                    f"{i+1}: {point}",
                    (10, 140 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                )

        # Handle board calibration mode
        if detector.is_calibrating:
            # Draw calibration points and lines
            detector.draw_calibration(display_frame)
        else:
            # Process the frame to detect chessboard squares
            display_frame = detector.process_frame(display_frame)
            if not detector.is_trained:
                # Display a message if the model is not trained
                cv2.putText(
                    display_frame,
                    "Model not trained",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
            else:
                # Predict the state of the chessboard
                predictions = detector.predict(frame)
                # Optionally, draw predictions on the board
                for i in range(8):
                    for j in range(8):
                        if predictions[i][j] == 1:
                            corners = detector.corners[i * 8 + j]
                            center_x = int(sum([pt[0] for pt in corners]) / 4)
                            center_y = int(sum([pt[1] for pt in corners]) / 4)
                            cv2.putText(
                                display_frame,
                                "x",
                                (center_x - 8, center_y + 8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2,
                            )

                # Detect movement and handle move detection
                movement = detector.detect_movement(frame)
                if movement:
                    print("Movement detected... waiting for piece to settle")
                    stable_count = 0
                    required_stable_frames = (
                        10  # Number of consecutive stable frames required
                    )
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                    move = detector.detect_move(frame)  # Define or assign a value to 'move'
                    if move:
                        # Execute the detected move
                        print(f"Move detected: {move}")
                        time.sleep(0.5)
                        if detector.execute_move(move):
                            print("Move executed successfully")
                            time.sleep(2)  # Allow time for the move to register
                            # Capture black's move
                            blacks_move_info = detector.capture_black_move()
                            if blacks_move_info:
                                from_square, to_square, piece, uci_move = (
                                    blacks_move_info
                                )
                                print("Black's move captured and processed")
                                print("Move info:", blacks_move_info)
                                # Generate trajectory for the robot arm
                                joint_trajectory = (
                                    robot_controller.generate_move_trajectory(uci_move)
                                )
                                if joint_trajectory:
                                    print("Joint trajectory generated successfully")
                                    # Display joint angles for debugging
                                    for i, joint_angles in enumerate(joint_trajectory):
                                        print(
                                            f"Point {i+1}: {np.round(joint_angles, 6)}"
                                        )

                                    # Serialize and publish the trajectory to qnode
                                    trajectory_json = json_trajectory(joint_trajectory)
                                    print(trajectory_json)
                                    rospy.loginfo(
                                        f"Publishing trajectory with {len(joint_trajectory)} points"
                                    )
                                    pub.publish(trajectory_json)
                                    print(f"Trajectory published to chess_moves topic")
                                    time.sleep(15)
                            else:
                                print("Failed to capture black's move")
                        else:
                            print("Invalid move detected")

        # Display the processed frame
        cv2.imshow("Chess Piece Detector", display_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            # Quit the program
            break
        elif key == ord("1"):
            # Capture frame with no pieces
            if not detector.is_trained:
                print("Capturing frame with no pieces...")
                empty_board = np.zeros((8, 8), dtype=int)
                detector.collect_training_data(frame, empty_board)
        elif key == ord("2"):
            # Capture frame with white pieces on the lower half
            if not detector.is_trained:
                print("Capturing frame with white pieces on lower half...")
                lower_half_board = np.zeros((8, 8), dtype=int)
                lower_half_board[4:8, :] = 1
                detector.collect_training_data(frame, lower_half_board)
        elif key == ord("3"):
            # Capture frame with black pieces on the upper half
            if not detector.is_trained:
                print("Capturing frame with black pieces on upper half...")
                upper_half_board = np.zeros((8, 8), dtype=int)
                upper_half_board[0:4, :] = 1
                detector.collect_training_data(frame, upper_half_board)
        elif key == ord("4"):
            # Capture frame with normal chess start
            if not detector.is_trained:
                print("Capturing frame with normal chess start...")
                normal_start_board = np.zeros((8, 8), dtype=int)
                normal_start_board[0:2, :] = 1
                normal_start_board[6:8, :] = 1
                detector.collect_training_data(frame, normal_start_board)
        elif key == ord("t"):
            # Train the model
            print("Training the model...")
            detector.train()
            print("Model trained successfully!")
        elif key == ord("s"):
            # Save the data
            filename = input("Enter filename to save data: ")
            detector.save_data(filename)
        elif key == ord("l"):
            # Load the data
            filename = input("Enter filename to load data: ")
            detector.load_data(filename)
            # Get the corners (this might be needed to regenerate them from the loaded points)
            detector.get_square_points()
            # Update robot controller with loaded corner positions
            if len(detector.corners) == 64:
                print("Updating robot controller with loaded board corners...")
                robot_controller.update_from_detector_corners(detector.corners)
            else:
                print(
                    f"Warning: Expected 64 corners but got {len(detector.corners)}. Robot controller not updated."
                )
        elif key == ord("a"):
            # Start Arena calibration mode
            detector.arena_calibration_mode = True
            detector.arena_controller.start_calibration()
        elif key == ord(" "):  # Space key
            # Add a corner during Arena calibration
            if detector.arena_controller.calibrating:
                if detector.arena_controller.add_corner():
                    print("Arena calibration complete!")
        elif key == 27:  # ESC key
            # Cancel Arena calibration
            if detector.arena_controller.calibrating:
                detector.arena_controller.calibrating = False
                detector.arena_controller.arena_corners = []
                print("Arena calibration cancelled")

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
