# This script implements forward and inverse kinematics for the Open Manipulator using ikpy.
# This is utilized for controlling the robot arm to reach specific positions and orientations in 3D space.
# Not currently in use this works together with the robot_chess_controller_interpolations class
import numpy as np
from ikpy.chain import Chain
from ikpy.utils import geometry
from ikpy.link import Link, URDFLink
import sympy as sp


# Function to convert a rotation matrix to a quaternion
def rotation_matrix_to_quaternion(R):
    # Compute the trace of the rotation matrix
    trace = np.trace(R)

    # Compute quaternion components based on the trace
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    return np.array([qx, qy, qz, qw])


# Function to convert a quaternion to a rotation matrix
def quaternion_to_rotation_matrix(q):
    x, y, z, w = q

    # Normalize the quaternion
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    # Compute the rotation matrix
    xx, xy, xz = x * x, x * y, x * z
    yy, yz, zz = y * y, y * z, z * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ]
    )

    return R


# Function to create the kinematic chain for the Open Manipulator
def create_open_manipulator_chain():
    # Define the chain using URDFLink for each joint and link
    chain = Chain(
        name="open_manipulator",
        links=[
            # Base link - fixed
            URDFLink(
                name="base_link",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 0],
            ),
            # Joint 1 - revolute
            URDFLink(
                name="joint1",
                origin_translation=[0.012, 0.0, 0.017],
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 1],
            ),
            # Joint 2 - revolute
            URDFLink(
                name="joint2",
                origin_translation=[0.0, 0.0, 0.0595],
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0],
            ),
            # Joint 3 - revolute
            URDFLink(
                name="joint3",
                origin_translation=[0.024, 0, 0.128],
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0],
            ),
            # Joint 4 - revolute
            URDFLink(
                name="joint4",
                origin_translation=[0.124, 0.0, 0.0],
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0],
            ),
            # End effector - fixed
            URDFLink(
                name="end_effector",
                origin_translation=[0.126, 0.0, 0.0],
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 0],
            ),
        ],
    )
    return chain


# Function to perform forward kinematics
def forward_kinematics_test(joint_angles):
    # Create the kinematic chain
    chain = create_open_manipulator_chain()

    # Perform forward kinematics
    fk_result = chain.forward_kinematics(joint_angles)
    position = fk_result[:3, 3]  # Extract the position
    orientation = rotation_matrix_to_quaternion(
        fk_result[:3, :3]
    )  # Convert to quaternion

    return {"position": position, "orientation": orientation}


# Function to perform inverse kinematics
def inverse_kinematics_test(target_position, target_orientation, initial_angles=None):
    # Create the kinematic chain
    chain = create_open_manipulator_chain()

    # Default initial angles if none provided
    if initial_angles is None:
        initial_angles = [0, 0, 0, 0, 0, 0]

    # Normalize the quaternion
    target_orientation = target_orientation / np.linalg.norm(target_orientation)

    # Convert quaternion to rotation matrix for IK
    target_orientation_matrix = quaternion_to_rotation_matrix(target_orientation)

    # Perform inverse kinematics
    new_angles = chain.inverse_kinematics(
        target_position=target_position,
        target_orientation=target_orientation_matrix,
        initial_position=initial_angles,
        orientation_mode="all",
    )

    return {
        "joint_angles": new_angles,
    }


# Example usage of forward and inverse kinematics
if __name__ == "__main__":
    print("Testing Forward Kinematics")

    # Test FK with specific joint angles
    test_angles = [
        0,
        0.594,
        -0.063,
        0.374,
        1.154,
        0,
    ]  # Base fixed + 4 revolute joints + end fixed
    fk_result = forward_kinematics_test(test_angles)

    print(
        "Joint Angles:", np.round(test_angles[1:5], 3)
    )  # Only show the 4 active joints
    print("End-effector Position:", np.round(fk_result["position"], 3))
    print(
        "End-effector Orientation (Quaternion):", np.round(fk_result["orientation"], 3)
    )

    print("\nTesting Inverse Kinematics")

    # Use the position and orientation from FK as targets for IK
    target_position = fk_result["position"]
    target_orientation = fk_result["orientation"]

    # Run IK to find joint angles
    ik_result = inverse_kinematics_test(
        target_position, target_orientation, test_angles
    )

    print("Target Position:", np.round(target_position, 3))
    print("Target Orientation:", np.round(target_orientation, 3))
    print(
        "Computed Joint Angles:", np.round(ik_result["joint_angles"][1:5], 3)
    )  # Only show active joints

    print("\nTesting Custom Target Pose")

    # Define a custom target pose
    custom_position = np.array([0.06322555, -0.09916947, 0.04768746])  # Custom position
    custom_orientation = np.array(
        [0.36301133, 0.59609231, -0.37249858, 0.61167111]
    )  # Custom orientation

    # Run IK for the custom pose
    custom_ik_result = inverse_kinematics_test(custom_position, custom_orientation)

    print("Custom Target Position:", np.round(custom_position, 3))
    print("Custom Target Orientation:", np.round(custom_orientation, 3))
    print(
        "Computed Joint Angles:", np.round(custom_ik_result["joint_angles"][1:5], 3)
    )  # Only show active joints
