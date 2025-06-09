# Robot Chess Controller
# This comes up with the trajectory of points the robot arm has to follow
# to move a piece from one square to another on a chessboard. Also accounting for piece captures and special moves
# All the angles are hardcoded in dictionaries
import numpy as np
import cv2
import rospy
from std_msgs.msg import Float64
from scipy.spatial.transform import Rotation, Slerp
import json


class RobotChessController:
    def __init__(self):
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

        # Captured piece storage configuration
        self.captured_piece_joints = [0, 1.154, 0.397, -0.186, 1.282, 0]
        self.captured_piece_joints2 = [0, 1.009, -0.120, -0.179, 1.284, 0]

        # Initialize the robot arm's joint angles for ground and hover positions
        self.ground_angles = {
            "a8": [0, -1.198039, -0.285, 0.506214, 1.267068, 0],
            "b8": [0, -1.066117, -0.435651, 0.655010, 1.325359, 0],
            "c8": [0, -0.903515, -0.635068, 0.802272, 1.397457, 0],
            "d8": [0, -0.526155, -0.848291, 0.900447, 1.483359, 0],
            "e8": [0, 0.242369, -0.928058, 0.929592, 1.497165, 0],
            "f8": [0, 0.734777, -0.872835, 0.901981, 1.431204, 0],
            "g8": [0, 1.121340, -0.504680, 0.774660, 1.282408, 0],
            "h8": [0, 1.176563, -0.365087, 0.585981, 1.306952, 0],
            "a7": [0, -0.994020, -0.145728, 0.368155, 1.306952, 0],
            "b7": [0, -0.813010, -0.228563, 0.507748, 1.329961, 0],
            "c7": [0, -0.593651, -0.418777, 0.658078, 1.351437, 0],
            "d7": [0, -0.280719, -0.558369, 0.774660, 1.354505, 0],
            "e7": [0, 0.141126, -0.612058, 0.794602, 1.371379, 0],
            "f7": [0, 0.497010, -0.589049, 0.727107, 1.371379, 0],
            "g7": [0, 0.779262, -0.437185, 0.633534, 1.339165, 0],
            "h7": [0, 0.990952, -0.061359, 0.386563, 1.323825, 0],
            "a6": [0, -0.822214, -0.006136, 0.219359, 1.316156, 0],
            "b6": [0, -0.676486, -0.038350, 0.316000, 1.356039, 0],
            "c6": [0, -0.452524, -0.148796, 0.420311, 1.362175, 0],
            "d6": [0, -0.199418, -0.309864, 0.605922, 1.287010, 0],
            "e6": [0, 0.070563, -0.342078, 0.619728, 1.287010, 0],
            "f6": [0, 0.411107, -0.316000, 0.610524, 1.264000, 0],
            "g6": [0, 0.648874, -0.119651, 0.420311, 1.323825, 0],
            "h6": [0, 0.826816, 0.019942, 0.239301, 1.323825, 0],
            "a5": [0, -0.694893, 0.127320, 0.036816, 1.345301, 0],
            "b5": [0, -0.546097, 0.116583, 0.133456, 1.362175, 0],
            "c5": [0, -0.363553, 0.013806, 0.273049, 1.316156, 0],
            "d5": [0, -0.156466, -0.075165, 0.371223, 1.288544, 0],
            "e5": [0, 0.072097, -0.116583, 0.398835, 1.262466, 0],
            "f5": [0, 0.297592, -0.111981, 0.369689, 1.262466, 0],
            "g5": [0, 0.501612, -0.038350, 0.286854, 1.262466, 0],
            "h5": [0, 0.671884, 0.041417, 0.168738, 1.270136, 0],
            "a4": [0, -0.602854, 0.351282, -0.280719, 1.468020, 0],
            "b4": [0, -0.470932, 0.311398, -0.148796, 1.437340, 0],
            "c4": [0, -0.299126, 0.216291, 0.007670, 1.360641, 0],
            "d4": [0, -0.124252, 0.138058, 0.101243, 1.325359, 0],
            "e4": [0, 0.076699, 0.084369, 0.148796, 1.302350, 0],
            "f4": [0, 0.271515, 0.096641, 0.116583, 1.340699, 0],
            "g4": [0, 0.418777, 0.161068, 0.027612, 1.314622, 0],
            "h4": [0, 0.566039, 0.230097, -0.073631, 1.320758, 0],
            "a3": [0, -0.512350, 0.481670, -0.504680, 1.487961, 0],
            "b3": [0, -0.388097, 0.497010, -0.524621, 1.570796, 0],
            "c3": [0, -0.245437, 0.477068, -0.391165, 1.529379, 0],
            "d3": [0, -0.087437, 0.411107, -0.328272, 1.512505, 0],
            "e3": [0, 0.050621, 0.391165, -0.291456, 1.468020, 0],
            "f3": [0, 0.193282, 0.311398, -0.179476, 1.380583, 0],
            "g3": [0, 0.362019, 0.312932, -0.194816, 1.342233, 0],
            "h3": [0, 0.500078, 0.437185, -0.412641, 1.437340, 0],
            "a2": [0, -0.458660, 0.780796, -1.021631, 1.695049, 0],
            "b2": [0, -0.343612, 0.740913, -0.846757, 1.612214, 0],
            "c2": [0, -0.234699, 0.604388, -0.613592, 1.501767, 0],
            "d2": [0, -0.076699, 0.581379, -0.592117, 1.529379, 0],
            "e2": [0, 0.067495, 0.559903, -0.587515, 1.529379, 0],
            "f2": [0, 0.200951, 0.556835, -0.612058, 1.549321, 0],
            "g2": [0, 0.323670, 0.645806, -0.783864, 1.609146, 0],
            "h2": [0, 0.441786, 0.751651, -0.987884, 1.696583, 0],
            "a1": [0, -0.398835, 0.918855, -1.282408, 1.655165, 0],
            "b1": [0, -0.282252, 0.915787, -1.280874, 1.711923, 0],
            "c1": [0, -0.177942, 0.897379, -1.244058, 1.741068, 0],
            "d1": [0, -0.064427, 0.897379, -1.247126, 1.784020, 0],
            "e1": [0, 0.067495, 0.909651, -1.271670, 1.785554, 0],
            "f1": [0, 0.177942, 0.905049, -1.287010, 1.785554, 0],
            "g1": [0, 0.294524, 0.924990, -1.314622, 1.785554, 0],
            "h1": [0, 0.408039, 0.974078, -1.385185, 1.761010, 0],
        }

        self.hover_angles = {
            "a8": [0, -1.251728, -0.240835, 0.030680, 1.793224, 0],
            "b8": [0, -1.168893, -0.506214, 0.271515, 1.773282, 0],
            "c8": [0, -0.984816, -0.780796, 0.349748, 1.876059, 0],
            "d8": [0, -0.562971, -0.880505, 0.371223, 1.998777, 0],
            "e8": [0, 0.567573, -0.923456, 0.383495, 1.997243, 0],
            "f8": [0, 0.842155, -0.742447, 0.389631, 1.869923, 0],
            "g8": [0, 1.194971, -0.520020, 0.269981, 1.814699, 0],
            "h8": [0, 1.365243, -0.309864, 0.111981, 1.764078, 0],
            "a7": [0, -1.058447, -0.202485, -0.015340, 1.736466, 0],
            "b7": [0, -0.882039, -0.377359, 0.122718, 1.790156, 0],
            "c7": [0, -0.559903, -0.592117, 0.292990, 1.790156, 0],
            "d7": [0, -0.254641, -0.642738, 0.389631, 1.799360, 0],
            "e7": [0, 0.418777, -0.699495, 0.314466, 1.857651, 0],
            "f7": [0, 0.719437, -0.584447, 0.222427, 1.842311, 0],
            "g7": [0, 0.960272, -0.404971, 0.056757, 1.842311, 0],
            "h7": [0, 1.075321, -0.222427, -0.069029, 1.761010, 0],
            "a6": [0, -0.866699, -0.125786, -0.158000, 1.761010, 0],
            "b6": [0, -0.645806, -0.200951, -0.055223, 1.785554, 0],
            "c6": [0, -0.400369, -0.371223, 0.154932, 1.742602, 0],
            "d6": [0, -0.108913, -0.377359, 0.131922, 1.773282, 0],
            "e6": [0, 0.230097, -0.377359, 0.118117, 1.773282, 0],
            "f6": [0, 0.423379, -0.391165, 0.187146, 1.690447, 0],
            "g6": [0, 0.751651, -0.277651, -0.073631, 1.790156, 0],
            "h6": [0, 0.900447, -0.082835, -0.305262, 1.807029, 0],
            "a5": [0, -0.750117, 0.059825, -0.377359, 1.791690, 0],
            "b5": [0, -0.497010, 0.058291, -0.375825, 1.837709, 0],
            "c5": [0, -0.322136, -0.007670, -0.296058, 1.837709, 0],
            "d5": [0, -0.099709, -0.079767, -0.259243, 1.836175, 0],
            "e5": [0, 0.139592, -0.079767, -0.230097, 1.836175, 0],
            "f5": [0, 0.369689, -0.079767, -0.283786, 1.837709, 0],
            "g5": [0, 0.339010, -0.092039, -0.078233, 1.779418, 0],
            "h5": [0, 0.766990, 0.134990, -0.579845, 1.839243, 0],
            "a4": [0, -0.610524, 0.400369, -0.888175, 1.960428, 0],
            "b4": [0, -0.466330, 0.256175, -0.575243, 1.849981, 0],
            "c4": [0, -0.299126, 0.144194, -0.369689, 1.744136, 0],
            "d4": [0, -0.092039, 0.087437, -0.354350, 1.744136, 0],
            "e4": [0, 0.082835, 0.075165, -0.185612, 1.670505, 0],
            "f4": [0, 0.279185, 0.064427, -0.138058, 1.629088, 0],
            "g4": [0, 0.472466, 0.038350, -0.357418, 1.742602, 0],
            "h4": [0, 0.610524, 0.220893, -0.619728, 1.802428, 0],
            "a3": [0, -0.524621, 0.547631, -1.012427, 1.920544, 0],
            "b3": [0, -0.389631, 0.500078, -0.941864, 1.908272, 0],
            "c3": [0, -0.234699, 0.366621, -0.702563, 1.840777, 0],
            "d3": [0, -0.070563, 0.228563, -0.553767, 1.761010, 0],
            "e3": [0, 0.115049, 0.220893, -0.562971, 1.756408, 0],
            "f3": [0, 0.268447, 0.220893, -0.572175, 1.756408, 0],
            "g3": [0, 0.420311, 0.234699, -0.644272, 1.756408, 0],
            "h3": [0, 0.562971, 0.374291, -0.845223, 1.768680, 0],
            "a2": [0, -0.469398, 0.681087, -1.314622, 1.946622, 0],
            "b2": [0, -0.328272, 0.693359, -1.368311, 1.981903, 0],
            "c2": [0, -0.193282, 0.694893, -1.354505, 2.012583, 0],
            "d2": [0, -0.062893, 0.694893, -1.293146, 2.015651, 0],
            "e2": [0, 0.047553, 0.696427, -1.294680, 2.015651, 0],
            "f2": [0, 0.210155, 0.696427, -1.297748, 2.015651, 0],
            "g2": [0, 0.378893, 0.711767, -1.329961, 2.015651, 0],
            "h2": [0, 0.510816, 0.725573, -1.443476, 2.015651, 0],
            "a1": [0, -0.411107, 0.826816, -1.580000, 1.872991, 0],
            "b1": [0, -0.314466, 0.840621, -1.538583, 1.900602, 0],
            "c1": [0, -0.182544, 0.834486, -1.540117, 1.952758, 0],
            "d1": [0, -0.058291, 0.816078, -1.543185, 2.021787, 0],
            "e1": [0, 0.082835, 0.816078, -1.601476, 2.027923, 0],
            "f1": [0, 0.185612, 0.816078, -1.601476, 2.057068, 0],
            "g1": [0, 0.332874, 0.822214, -1.632156, 2.043262, 0],
            "h1": [0, 0.441786, 0.832952, -1.632156, 1.972699, 0],
        }

        try:

            self.is_initialized = True
            print("Robot control initialized successfully")
        except ImportError:
            self.is_initialized = False
            print("Robot control disabled")
        except Exception as e:
            self.is_initialized = False
            print(f"Robot control disabled - Error: {e}")

    # Generate a trajectory for moving a piece from one square to another
    # Handles standard moves, captures, castling, and en passant
    def generate_move_trajectory(self, uci_move):

        print(f"Generating move trajectory from {uci_move}")
        if not self.is_initialized:
            print("Robot controller not initialized")
            return None

        # Parse UCI move string (e.g. "d7d5" or "d7d5x" for captures)
        from_square = uci_move[:2]
        to_square = uci_move[2:4]  # Get destination before the 'x' if it exists
        is_capture = uci_move.endswith("x")

        # Check if we have hardcoded positions for these squares
        if from_square not in self.ground_angles or to_square not in self.ground_angles:
            print(f"Square position not found for {from_square} or {to_square}")
            return None

        if from_square not in self.hover_angles or to_square not in self.hover_angles:
            print(f"Hover position not found for {from_square} or {to_square}")
            return None

        # Check for special moves: castling
        is_kingside_castling = from_square == "e8" and to_square == "g8"
        is_queenside_castling = from_square == "e8" and to_square == "c8"

        if is_kingside_castling:
            print(
                "Detected BLACK KINGSIDE CASTLING (e8g8) - Need to also move rook h8->f8"
            )
            rook_from = "h8"
            rook_to = "f8"
            # Verify we have positions for the rook squares
            if rook_from not in self.ground_angles or rook_to not in self.ground_angles:
                print(f"Error: Missing angles for rook movement {rook_from}->{rook_to}")
                return None
        elif is_queenside_castling:
            print(
                "Detected BLACK QUEENSIDE CASTLING (e8c8) - Need to also move rook a8->d8"
            )
            rook_from = "a8"
            rook_to = "d8"
            # Verify we have positions for the rook squares
            if rook_from not in self.ground_angles or rook_to not in self.ground_angles:
                print(f"Error: Missing angles for rook movement {rook_from}->{rook_to}")
                return None

        # Initialize trajectory
        joint_trajectory = []

        # For castling, we need to move both the king and rook
        if is_kingside_castling or is_queenside_castling:
            # ===== STEP 1: Move King first =====
            print(f"1. Moving King from {from_square} to {to_square}")

            # 1.1. Hover above king
            hover_king_angles = self.hover_angles[from_square].copy()
            hover_king_angles[5] = 0.5  # Open gripper
            joint_trajectory.append(hover_king_angles)
            print(
                f"  Step 1.1: Hover above king at {from_square}, angles: {np.round(hover_king_angles[1:5], 3)}"
            )

            # 1.2. Lower to grab king
            ground_king_angles = self.ground_angles[from_square].copy()
            ground_king_angles[5] = 0.0  # Close gripper
            joint_trajectory.append(ground_king_angles)
            print(
                f"  Step 1.2: Grab king at {from_square}, angles: {np.round(ground_king_angles[1:5], 3)}"
            )

            # 1.3. Raise with king
            raise_king_angles = self.hover_angles[from_square].copy()
            raise_king_angles[5] = 0.0  # Keep gripper closed
            joint_trajectory.append(raise_king_angles)
            print(
                f"  Step 1.3: Raise king above {from_square}, angles: {np.round(raise_king_angles[1:5], 3)}"
            )

            # 1.4. Hover above king destination
            hover_king_dest_angles = self.hover_angles[to_square].copy()
            hover_king_dest_angles[5] = 0.0  # Keep gripper closed
            joint_trajectory.append(hover_king_dest_angles)
            print(
                f"  Step 1.4: Hover above king destination {to_square}, angles: {np.round(hover_king_dest_angles[1:5], 3)}"
            )

            # 1.5. Lower king to destination
            place_king_angles = self.ground_angles[to_square].copy()
            place_king_angles[5] = 0.0  # Keep gripper closed until placed
            joint_trajectory.append(place_king_angles)
            print(
                f"  Step 1.5: Place king at {to_square}, angles: {np.round(place_king_angles[1:5], 3)}"
            )

            # 1.6. Raise after placing king
            final_king_hover_angles = self.hover_angles[to_square].copy()
            final_king_hover_angles[5] = 0.5  # Open gripper
            joint_trajectory.append(final_king_hover_angles)
            print(
                f"  Step 1.6: Raise above king destination {to_square}, angles: {np.round(final_king_hover_angles[1:5], 3)}"
            )

            # ===== STEP 2: Move Rook next =====
            print(f"2. Moving Rook from {rook_from} to {rook_to}")

            # 2.1. Hover above rook
            hover_rook_angles = self.hover_angles[rook_from].copy()
            hover_rook_angles[5] = 0.5  # Open gripper
            joint_trajectory.append(hover_rook_angles)
            print(
                f"  Step 2.1: Hover above rook at {rook_from}, angles: {np.round(hover_rook_angles[1:5], 3)}"
            )

            # 2.2. Lower to grab rook
            ground_rook_angles = self.ground_angles[rook_from].copy()
            ground_rook_angles[5] = 0.0  # Close gripper
            joint_trajectory.append(ground_rook_angles)
            print(
                f"  Step 2.2: Grab rook at {rook_from}, angles: {np.round(ground_rook_angles[1:5], 3)}"
            )

            # 2.3. Raise with rook
            raise_rook_angles = self.hover_angles[rook_from].copy()
            raise_rook_angles[5] = 0.0  # Keep gripper closed
            joint_trajectory.append(raise_rook_angles)
            print(
                f"  Step 2.3: Raise rook above {rook_from}, angles: {np.round(raise_rook_angles[1:5], 3)}"
            )

            # 2.4. Hover above rook destination
            hover_rook_dest_angles = self.hover_angles[rook_to].copy()
            hover_rook_dest_angles[5] = 0.0  # Keep gripper closed
            joint_trajectory.append(hover_rook_dest_angles)
            print(
                f"  Step 2.4: Hover above rook destination {rook_to}, angles: {np.round(hover_rook_dest_angles[1:5], 3)}"
            )

            # 2.5. Lower rook to destination
            place_rook_angles = self.ground_angles[rook_to].copy()
            place_rook_angles[5] = 0.0  # Keep gripper closed until placed
            joint_trajectory.append(place_rook_angles)
            print(
                f"  Step 2.5: Place rook at {rook_to}, angles: {np.round(place_rook_angles[1:5], 3)}"
            )

            # 2.6. Raise after placing rook
            final_rook_hover_angles = self.hover_angles[rook_to].copy()
            final_rook_hover_angles[5] = 0.5  # Open gripper
            joint_trajectory.append(final_rook_hover_angles)
            print(
                f"  Step 2.6: Raise above rook destination {rook_to}, angles: {np.round(final_rook_hover_angles[1:5], 3)}"
            )

            print(f"Castling trajectory completed with {len(joint_trajectory)} points")
            return joint_trajectory

        # For capture moves, we need to remove the captured piece first
        if is_capture:
            # 1. Hover above destination square
            hover_to_angles = self.hover_angles[to_square].copy()
            hover_to_angles[5] = 0.5  # Open gripper
            joint_trajectory.append(hover_to_angles)
            print(
                f"Point 1: Hover above {to_square}, Joint angles: {np.round(hover_to_angles[1:5], 3)}"
            )

            # 2. Lower to grab captured piece
            ground_to_angles = self.ground_angles[to_square].copy()
            ground_to_angles[5] = 0.0  # Close gripper
            joint_trajectory.append(ground_to_angles)
            print(
                f"Point 2: Grab piece at {to_square}, Joint angles: {np.round(ground_to_angles[1:5], 3)}"
            )

            # 3. Raise with captured piece
            raise_angles = self.hover_angles[to_square].copy()
            raise_angles[5] = 0.0  # Keep gripper closed
            joint_trajectory.append(raise_angles)
            print(
                f"Point 3: Raise above {to_square}, Joint angles: {np.round(raise_angles[1:5], 3)}"
            )

            # 4. Move to captured piece storage
            storage1_angles = self.captured_piece_joints2.copy()
            storage1_angles[5] = 0.0  # Keep gripper closed
            joint_trajectory.append(storage1_angles)
            print(
                f"Point 4: Position: [Storage 1], Joint angles: {np.round(storage1_angles[1:5], 3)}"
            )

            # 5. Move to captured piece final storage
            storage2_angles = self.captured_piece_joints.copy()
            storage2_angles[5] = 0.0  # Keep gripper closed
            joint_trajectory.append(storage2_angles)
            print(
                f"Point 5: Position: [Storage 2], Joint angles: {np.round(storage2_angles[1:5], 3)}"
            )

            # 6. Return to intermediate storage position
            joint_trajectory.append(storage1_angles)
            print(
                f"Point 6: Position: [Storage 1], Joint angles: {np.round(storage1_angles[1:5], 3)}"
            )

        # Move the piece from source to destination
        # 7. Hover above source square
        hover_from_angles = self.hover_angles[from_square].copy()
        hover_from_angles[5] = 0.5  # Open gripper
        joint_trajectory.append(hover_from_angles)
        print(
            f"Point {len(joint_trajectory)}: Hover above {from_square}, Joint angles: {np.round(hover_from_angles[1:5], 3)}"
        )

        # 8. Lower to grab piece
        ground_from_angles = self.ground_angles[from_square].copy()
        ground_from_angles[5] = 0.0  # Close gripper
        joint_trajectory.append(ground_from_angles)
        print(
            f"Point {len(joint_trajectory)}: Grab piece at {from_square}, Joint angles: {np.round(ground_from_angles[1:5], 3)}"
        )

        # 9. Raise with piece
        raise_from_angles = self.hover_angles[from_square].copy()
        raise_from_angles[5] = 0.0  # Keep gripper closed
        joint_trajectory.append(raise_from_angles)
        print(
            f"Point {len(joint_trajectory)}: Raise above {from_square}, Joint angles: {np.round(raise_from_angles[1:5], 3)}"
        )

        # 10. Hover above destination
        hover_to_angles = self.hover_angles[to_square].copy()
        hover_to_angles[5] = 0.0  # Keep gripper closed
        joint_trajectory.append(hover_to_angles)
        print(
            f"Point {len(joint_trajectory)}: Hover above {to_square}, Joint angles: {np.round(hover_to_angles[1:5], 3)}"
        )

        # 11. Lower to place piece
        place_angles = self.ground_angles[to_square].copy()
        place_angles[5] = 0.0  # Keep gripper closed until placed
        joint_trajectory.append(place_angles)
        print(
            f"Point {len(joint_trajectory)}: Place piece at {to_square}, Joint angles: {np.round(place_angles[1:5], 3)}"
        )

        # 12. Raise after placing
        final_hover_angles = self.hover_angles[to_square].copy()
        final_hover_angles[5] = 0.5  # Open gripper
        joint_trajectory.append(final_hover_angles)
        print(
            f"Point {len(joint_trajectory)}: Raise above {to_square}, Joint angles: {np.round(final_hover_angles[1:5], 3)}"
        )

        print(f"Total trajectory points: {len(joint_trajectory)}")
        return joint_trajectory


# Function to convert the joint trajectory to JSON format
def json_trajectory(joint_trajectory):
    # Convert numpy arrays to lists for JSON serialization
    trajectory_list = []
    for angles in joint_trajectory:
        # Convert each point to a list with clean float values (limited decimal places)
        angles_list = [float(f"{val:.6f}") for val in angles]
        trajectory_list.append(angles_list)
    return json.dumps(trajectory_list)
