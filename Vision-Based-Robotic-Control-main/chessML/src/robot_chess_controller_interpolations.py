# This Class is not currently in use
# Robot arm controller through interpolation of joint angles across chess board utilizing bilinear interpolation and SLERP

# Import necessary libraries
import rospy
from std_msgs.msg import Float64
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import cv2


class RobotChessController:
    def __init__(self):
        # Initialize the robot controller
        self.is_initialized = False  # Check if IKFK is available

        # Initialize ROS publishers for joint control
        self.joint1_pub = rospy.Publisher(
            "/joint1_position/command", Float64, queue_size=10
        )
        self.joint2_pub = rospy.Publisher(
            "/joint2_position/command", Float64, queue_size=10
        )
        self.joint3_pub = rospy.Publisher(
            "/joint3_position/command", Float64, queue_size=10
        )
        self.joint4_pub = rospy.Publisher(
            "/joint4_position/command", Float64, queue_size=10
        )
        self.gripper_pub = rospy.Publisher(
            "/gripper_position/command", Float64, queue_size=10
        )

        # Hard-coded joint angles for calibrated positions
        self.calibrated_positions = {
            # Corners of the chessboard
            "a8": [0, -1.230, -0.236, 0.571, 1.278, 0],  # Top-left corner
            "h8": [0, 1.269, -0.199, 0.523, 1.261, 0],  # Top-right corner
            "h1": [0, 0.422, 0.798, -0.977, 1.470, 0],  # Bottom-right corner
            "a1": [0, -0.393, 0.816, -0.999, 1.470, 0],  # Bottom-left corner
            # Middle edges and other positions
            "d8": [0, -0.282, -0.962, 0.983, 1.559, 0],  # Middle top edge
            "d1": [0, -0.051, 0.683, -0.761, 1.482, 0],  # Middle bottom edge
            "a4": [0, -0.584, 0.451, -0.336, 1.571, 0],  # Middle left edge
            "h4": [0, 0.621, 0.465, -0.385, 1.571, 0],  # Middle right edge
        }

        # Store camera positions for all squares
        self.square_centers = {}

        # World positions corresponding to corners (calculated using FK)
        self.world_positions = {}
        self.calibrated_orientations = {}

        # Dictionary to store calculated positions for all squares
        self.chess_square_positions = {}

        # Parameters for movement
        self.hover_offset = 0.07  # Hover height above the board surface

        # Captured piece storage configuration
        self.captured_piece_joints = [
            0,
            1.154,
            0.397,
            -0.186,
            1.282,
            0,
        ]  # Storage position 1
        self.captured_piece_joints2 = [
            0,
            1.009,
            -0.120,
            -0.179,
            1.284,
            0,
        ]  # Storage position 2

        # Try to import IKFK module for kinematics
        try:
            from IKFK import (
                create_open_manipulator_chain,
                forward_kinematics_test,
                inverse_kinematics_test,
            )

            self.chain = create_open_manipulator_chain()
            self.forward_kinematics = forward_kinematics_test
            self.inverse_kinematics = inverse_kinematics_test
            self.is_initialized = True
            print("Robot controller initialized successfully")

            # Calculate world positions for the four corners immediately after initialization
            if self.calculate_world_positions():
                print("Successfully calculated world positions for corners")
            else:
                print("Failed to calculate world positions for corners")
        except ImportError as e:
            print(f"Robot control disabled: {e}")

    def update_from_detector_corners(self, detector_corners):
        """
        Update square centers using the corner points from ChessPieceDetector.

        Args:
            detector_corners: The corners array from ChessPieceDetector, containing
                              corner coordinates for all 64 squares.
        """
        if len(detector_corners) != 64:
            print(f"Error: Expected 64 squares, got {len(detector_corners)}")
            return False

        # Clear previous centers
        self.square_centers = {}

        # Files and ranks for chess notation
        files = "abcdefgh"
        ranks = "87654321"

        # Calculate center point of each square
        for i in range(8):
            for j in range(8):
                square_idx = i * 8 + j
                corners = detector_corners[square_idx]

                # Calculate center as average of four corners
                center_x = sum(corner[0] for corner in corners) / 4
                center_y = sum(corner[1] for corner in corners) / 4

                # Convert to chess notation
                square = files[j] + ranks[i]

                # Store the center point
                self.square_centers[square] = (center_x, center_y)

        print(f"Updated center points for all 64 squares from detector corners")

        # Map important corners to our world positions
        if self.is_initialized:
            self.map_camera_to_world()

        return True

    def calculate_world_positions(self):
        """Calculate world coordinates from calibrated joint angles."""
        if not self.is_initialized:
            return False

        for square, angles in self.calibrated_positions.items():
            print(f"Calculating world position for {square} with angles {angles}")
            fk_result = self.forward_kinematics(angles)
            self.world_positions[square] = fk_result["position"]
            self.calibrated_orientations[square] = fk_result["orientation"]

        print("Corner world positions calculated from joint angles:")
        for square, pos in self.world_positions.items():
            print(f"  {square}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

        return True

    def map_camera_to_world(self):
        """Map camera center points to world positions using our calibrated corners."""
        if not self.is_initialized:
            return False

        if len(self.square_centers) != 64:
            print("Square centers not fully populated")
            return False

        # Get the camera center points
        camera_centers = {
            "a8": self.square_centers.get("a8"),
            "h8": self.square_centers.get("h8"),
            "h1": self.square_centers.get("h1"),
            "a1": self.square_centers.get("a1"),
        }

        if None in camera_centers.values():
            print("Missing camera center positions")
            return False

        # Convert world and camera positions to numpy arrays for easier computation
        world_corners = np.array(
            [
                self.world_positions["a8"],  # Top-left
                self.world_positions["h8"],  # Top-right
                self.world_positions["h1"],  # Bottom-right
                self.world_positions["a1"],  # Bottom-left
            ]
        )

        camera_corners_array = np.array(
            [
                camera_centers["a8"],  # Top-left
                camera_centers["h8"],  # Top-right
                camera_centers["h1"],  # Bottom-right
                camera_centers["a1"],  # Bottom-left
            ]
        )

        # Get corner orientations
        corner_orientations = np.array(
            [
                self.calibrated_orientations["a8"],  # Top-left
                self.calibrated_orientations["h8"],  # Top-right
                self.calibrated_orientations["h1"],  # Bottom-right
                self.calibrated_orientations["a1"],  # Bottom-left
            ]
        )

        # For each square, calculate its world position
        files = "abcdefgh"
        ranks = "87654321"

        for rank in ranks:
            for file in files:
                square = file + rank
                if square in self.square_centers:
                    camera_pos = self.square_centers[square]
                    x_norm, y_norm = self._get_normalized_position(
                        camera_pos, camera_corners_array
                    )

                    # Get interpolated position and orientation
                    world_pos = self._bilinear_interpolate(
                        world_corners, x_norm, y_norm
                    )
                    orientation = self._interpolate_orientation(
                        corner_orientations, x_norm, y_norm
                    )

                    # Store both position and orientation
                    self.chess_square_positions[square] = {
                        "camera": camera_pos,
                        "world": world_pos,
                        "orientation": orientation,
                    }
                    print(
                        f"Mapped {square} to {world_pos} with an orientation of {orientation}"
                    )

        print(f"Mapped camera centers to world positions for all 64 squares")
        return True

    def _get_normalized_position(self, point, corners):
        """
        Calculate the normalized position (0-1) of a point within the quadrilateral
        defined by the four corners using a perspective transform.

        Args:
            point: The (x, y) point to normalize
            corners: Array of four corner points [top_left, top_right, bottom_right, bottom_left]

        Returns:
            (x_norm, y_norm): Normalized position using perspective transform
        """
        # Define the destination points for a normalized square
        dst_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)

        # Convert corners to float32 array
        src_points = corners.astype(np.float32)

        # Calculate the perspective transform matrix
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Convert point to homogeneous coordinates
        point_h = np.array([[[point[0], point[1]]]], dtype=np.float32)

        # Apply perspective transform to get normalized coordinates
        normalized = cv2.perspectiveTransform(point_h, perspective_matrix)
        x_norm, y_norm = normalized[0][0]

        # Constrain to 0-1 range to handle numerical errors
        x_norm = max(0, min(1, x_norm))
        y_norm = max(0, min(1, y_norm))

        return x_norm, y_norm

    def _bilinear_interpolate(self, corners, x_norm, y_norm):
        """
        Perform bilinear interpolation between four corner points.

        Args:
            corners: Array of four points [top_left, top_right, bottom_right, bottom_left]
            x_norm: Normalized x position (0-1)
            y_norm: Normalized y position (0-1)

        Returns:
            Interpolated position
        """
        # Linear interpolation along top and bottom edges
        top = corners[0] * (1 - x_norm) + corners[1] * x_norm
        bottom = corners[3] * (1 - x_norm) + corners[2] * x_norm

        # Linear interpolation between top and bottom
        return top * (1 - y_norm) + bottom * y_norm

    def _interpolate_orientation(self, corners_orient, x_norm, y_norm):
        """
        Interpolate orientation using scipy's Slerp.

        Args:
            corners_orient: Array of quaternions for the four corners
            x_norm: Normalized x position (0-1)
            y_norm: Normalized y position (0-1)

        Returns:
            Interpolated quaternion
        """
        # Create Rotation objects from quaternions for each corner
        rotations = Rotation.from_quat(corners_orient)

        # First interpolate along top and bottom edges
        top_edge = Slerp([0, 1], rotations[[0, 1]])
        bottom_edge = Slerp([0, 1], rotations[[3, 2]])

        # Get the rotations at x_norm position
        top_rot = top_edge(x_norm)
        bottom_rot = bottom_edge(x_norm)

        # Now interpolate between top and bottom
        vertical_rots = Rotation.from_quat([top_rot.as_quat(), bottom_rot.as_quat()])
        vertical_interp = Slerp([0, 1], vertical_rots)

        final_rot = vertical_interp(y_norm)

        return final_rot.as_quat()

    def get_square_camera_position(self, square):
        """Get camera position for a chess square"""
        if square in self.square_centers:
            return self.square_centers[square]
        return None

    def get_square_world_position(self, square):
        """Get world position for a chess square"""
        if square in self.chess_square_positions:
            return self.chess_square_positions[square]["world"]
        return None

    def generate_move_trajectory(self, uci_move):
        """
        Generate joint angles for moving a piece from one square to another.
        Uses world positions for hover points and calibrated positions for final positions when available.
        Returns a list of positions that can be sent as a message.
        """
        print(f"Generating move trajectory from {uci_move}")
        if not self.is_initialized:
            print("Robot controller not initialized")
            return None

        # Parse UCI move string (e.g. "d7d5" or "d7d5x" for captures)
        from_square = uci_move[:2]
        to_square = uci_move[2:4]  # Get destination before the 'x' if it exists

        # Check if we have calibrated positions for these squares
        use_calibrated_from = from_square in self.calibrated_positions
        use_calibrated_to = to_square in self.calibrated_positions

        # Log which positions we're using
        if use_calibrated_from:
            print(f"Using hardcoded calibrated position for {from_square}")
        if use_calibrated_to:
            print(f"Using hardcoded calibrated position for {to_square}")

        # Ensure we have positions for both squares (either calibrated or interpolated)
        if not use_calibrated_from and from_square not in self.chess_square_positions:
            print(f"Square position not calculated for {from_square}")
            return None

        if not use_calibrated_to and to_square not in self.chess_square_positions:
            print(f"Square position not calculated for {to_square}")
            return None

        # Get world positions for both squares - needed for hover calculations
        # For calibrated squares, get world positions from FK result stored in world_positions
        if use_calibrated_from:
            from_pos = self.world_positions[from_square]
            from_orientation = self.calibrated_orientations[from_square]
        else:
            from_pos = self.chess_square_positions[from_square]["world"]
            from_orientation = self.chess_square_positions[from_square]["orientation"]

        if use_calibrated_to:
            to_pos = self.world_positions[to_square]
            to_orientation = self.calibrated_orientations[to_square]
        else:
            to_pos = self.chess_square_positions[to_square]["world"]
            to_orientation = self.chess_square_positions[to_square]["orientation"]

        # Create hover positions - always use world coordinates + offset
        hover_from = np.copy(from_pos)
        hover_from[2] += self.hover_offset
        hover_to = np.copy(to_pos)
        hover_to[2] += self.hover_offset

        # Initialize trajectory
        joint_trajectory = []

        # For capture moves
        if uci_move.endswith("x"):
            # Get initial solution for hovering above destination (capture piece)
            initial_hover_ik = self.inverse_kinematics(hover_to, to_orientation)
            hover_angles = initial_hover_ik["joint_angles"].copy()
            hover_angles[5] = 0.5  # Open gripper
            joint_trajectory.append(hover_angles)
            print(
                f"Point 1: Position: {hover_to}, Joint angles: {np.round(hover_angles[1:5], 3)}"
            )

            # Lower to grab captured piece - use calibrated position if available
            if use_calibrated_to:
                grab_angles = self.calibrated_positions[to_square].copy()
                grab_angles[5] = 0.0  # Close gripper
            else:
                grab_ik = self.inverse_kinematics(to_pos, to_orientation, hover_angles)
                grab_angles = grab_ik["joint_angles"].copy()
                grab_angles[5] = 0.0  # Close gripper

            joint_trajectory.append(grab_angles)
            print(
                f"Point 2: Position: {to_pos}, Joint angles: {np.round(grab_angles[1:5], 3)}"
            )

            # Raise with captured piece - back to hover position
            raise_angles = hover_angles.copy()
            raise_angles[5] = 0.0  # Keep gripper closed
            joint_trajectory.append(raise_angles)
            print(
                f"Point 3: Position: {hover_to}, Joint angles: {np.round(raise_angles[1:5], 3)}"
            )

            # Move to captured piece storage (using pre-calibrated values)
            storage1_angles = self.captured_piece_joints2.copy()
            storage1_angles[5] = 0.0  # Keep gripper closed
            joint_trajectory.append(storage1_angles)
            print(
                f"Point 4: Position: [Storage 1], Joint angles: {np.round(storage1_angles[1:5], 3)} (pre-calibrated)"
            )

            storage2_angles = self.captured_piece_joints.copy()
            storage2_angles[5] = 0.0  # Keep gripper closed
            joint_trajectory.append(storage2_angles)
            print(
                f"Point 5: Position: [Storage 2], Joint angles: {np.round(storage2_angles[1:5], 3)} (pre-calibrated)"
            )

            joint_trajectory.append(storage1_angles)
            print(
                f"Point 6: Position: [Storage 1], Joint angles: {np.round(storage1_angles[1:5], 3)} (pre-calibrated)"
            )

        # Get hover position above source square - always use IK for hovering
        source_hover_ik = self.inverse_kinematics(hover_from, from_orientation)
        source_hover_angles = source_hover_ik["joint_angles"].copy()
        source_hover_angles[5] = 0.5  # Open gripper
        joint_trajectory.append(source_hover_angles)
        print(
            f"Point {len(joint_trajectory)}: Hover Position: {hover_from}, Joint angles: {np.round(source_hover_angles[1:5], 3)}"
        )

        # Lower to grab piece - use calibrated position if available
        if use_calibrated_from:
            source_grab_angles = self.calibrated_positions[from_square].copy()
            source_grab_angles[5] = 0.0  # Close gripper
        else:
            source_grab_ik = self.inverse_kinematics(
                from_pos, from_orientation, source_hover_angles
            )
            source_grab_angles = source_grab_ik["joint_angles"].copy()
            source_grab_angles[5] = 0.0  # Close gripper

        joint_trajectory.append(source_grab_angles)
        print(
            f"Point {len(joint_trajectory)}: Source Position: {from_pos}, Joint angles: {np.round(source_grab_angles[1:5], 3)}"
        )

        # Raise with piece - back to hover position
        source_raise_angles = source_hover_angles.copy()
        source_raise_angles[5] = 0.0  # Keep gripper closed
        joint_trajectory.append(source_raise_angles)
        print(
            f"Point {len(joint_trajectory)}: Hover Position: {hover_from}, Joint angles: {np.round(source_raise_angles[1:5], 3)}"
        )

        # Move to hover above destination - always use IK
        if uci_move.endswith("x") and np.array_equal(hover_to, hover_from):
            # Use the same solution as when approaching source
            dest_hover_angles = source_hover_angles.copy()
            dest_hover_angles[5] = 0.0  # Keep gripper closed
        elif uci_move.endswith("x") and np.array_equal(to_pos, from_pos):
            # If it's the same exact position, reuse the hover angles from capture
            dest_hover_angles = hover_angles.copy()
            dest_hover_angles[5] = 0.0  # Keep gripper closed
        else:
            # Get solution for destination hover position
            dest_hover_ik = self.inverse_kinematics(
                hover_to, to_orientation, source_raise_angles
            )
            dest_hover_angles = dest_hover_ik["joint_angles"].copy()
            dest_hover_angles[5] = 0.0  # Keep gripper closed

        joint_trajectory.append(dest_hover_angles)
        print(
            f"Point {len(joint_trajectory)}: Hover Position: {hover_to}, Joint angles: {np.round(dest_hover_angles[1:5], 3)}"
        )

        # Lower to place piece - use calibrated position if available
        if use_calibrated_to:
            dest_place_angles = self.calibrated_positions[to_square].copy()
            dest_place_angles[5] = 0.0  # Keep gripper closed until placed
        else:
            dest_place_ik = self.inverse_kinematics(
                to_pos, to_orientation, dest_hover_angles
            )
            dest_place_angles = dest_place_ik["joint_angles"].copy()
            dest_place_angles[5] = 0.0  # Keep gripper closed until placed

        joint_trajectory.append(dest_place_angles)
        print(
            f"Point {len(joint_trajectory)}: Destination Position: {to_pos}, Joint angles: {np.round(dest_place_angles[1:5], 3)}"
        )

        # Raise after placing - back to hover position
        dest_raise_angles = dest_hover_angles.copy()
        dest_raise_angles[5] = 0.5  # Open gripper
        joint_trajectory.append(dest_raise_angles)
        print(
            f"Point {len(joint_trajectory)}: Hover Position: {hover_to}, Joint angles: {np.round(dest_raise_angles[1:5], 3)}"
        )

        print(f"Total trajectory points: {len(joint_trajectory)}")
        return joint_trajectory
