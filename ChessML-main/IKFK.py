import numpy as np
from ikpy.chain import Chain
from ikpy.utils import geometry
from ikpy.link import Link, URDFLink
import sympy as sp


def rotation_matrix_to_quaternion(R):
    
    #Convert a rotation matrix to a quaternion.
    # r = 3x3 rotation matrix
    #Returns: quaternion [x, y, z, w]

    trace = np.trace(R)
    
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


def quaternion_to_rotation_matrix(q):
    #Convert a quaternion to a rotation matrix.
    # q = quaternion in format [x, y, z, w]   
    # Returns: 3x3 rotation matrix
    
    x, y, z, w = q
    
    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    
    # Convert quaternion to rotation matrix
    xx, xy, xz = x*x, x*y, x*z
    yy, yz, zz = y*y, y*z, z*z
    wx, wy, wz = w*x, w*y, w*z
    
    R = np.array([
        [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)],
        [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)]
    ])
    
    return R


def create_open_manipulator_chain():
    #kinametic chain for open manipulator spcefic links from urdf file
    chain = Chain(name="open_manipulator", links=[
        # Base link - fixed
        URDFLink(
            name="base_link",
            origin_translation=[0, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 0]
        ),
        # Joint 1 - revolute
        URDFLink(
            name="joint1",
            origin_translation=[0.012, 0.0, 0.017],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 1]
        ),
        # Joint 2 - revolute
        URDFLink(
            name="joint2",
            origin_translation=[0.0, 0.0, 0.0595],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0]
        ),
        # Joint 3 - revolute
        URDFLink(
            name="joint3",
            origin_translation=[0.024, 0, 0.128],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0]
        ),
        # Joint 4 - revolute
        URDFLink(
            name="joint4",
            origin_translation=[0.124, 0.0, 0.0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0]
        ),
        # End effector - fixed
        URDFLink(
            name="end_effector",
            origin_translation=[0.126, 0.0, 0.0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 0]
        )
    ])
    return chain


def forward_kinematics_test(joint_angles):
    # perform forward kinematics to get end-effector position and orientation.
    # joint_angles: List of joint angles [base, joint1, joint2, joint3, joint4, end_effector]
    # the base and end_effector values should be 0 (fixed joints)
    # Returns a dict containing the end-effector position and orientation
 
    chain = create_open_manipulator_chain()
    
    # Perform FK
    fk_result = chain.forward_kinematics(joint_angles)
    position = fk_result[:3, 3]  # Extract the position
    orientation = rotation_matrix_to_quaternion(fk_result[:3, :3])  # Convert to quaternion
    
    return {
        "position": position,
        "orientation": orientation
    }


def inverse_kinematics_test(target_position, target_orientation, initial_angles=None):
    # perform inverse kinematics to get joint angles for a desired end-effector pose.
    
    # target_position: Desired end-effector position [x, y, z]
    # target_orientation: Desired end-effector orientation as quaternion [x, y, z, w]
    # initial_angles: Initial guess for joint angles is optional
    # Returns: A dictionary containing the joint angles and error metrics are commented out for performance in main py
    
    chain = create_open_manipulator_chain()
    
    # Default initial angles if none provided
    if initial_angles is None:
        initial_angles = [0, 0, 0, 0, 0, 0]
    
    # Normalize the quaternion
    target_orientation = target_orientation / np.linalg.norm(target_orientation)

    # Convert quaternion to rotation matrix for IK
    target_orientation_matrix = quaternion_to_rotation_matrix(target_orientation)

    # Perform IK
    new_angles = chain.inverse_kinematics(
        target_position=target_position,
        target_orientation=target_orientation_matrix,
        initial_position=initial_angles,
        orientation_mode="all"
    )
    
    # # Verify the solution with FK
    # fk_result = chain.forward_kinematics(new_angles)
    # verified_position = fk_result[:3, 3]
    # verified_orientation = rotation_matrix_to_quaternion(fk_result[:3, :3])

    # # Compute errors
    # position_error = np.linalg.norm(np.array(target_position) - np.array(verified_position))
    # orientation_error = np.linalg.norm(np.array(target_orientation) - np.array(verified_orientation))


    #return new_angles
    return {
        "joint_angles": new_angles,
    #    "verified_position": verified_position,
    #    "verified_orientation": verified_orientation,
    #    "position_error": position_error,
    #    "orientation_error": orientation_error
    }

# h5 - 0.594, -0.063, 0.374, 1.154

# Example usage
if __name__ == "__main__":
    print("Testing Forward Kinematics")
    
    # Test FK with specific joint angles
    test_angles = [0, 0.594, -0.063, 0.374, 1.154,0]  # Base fixed + 4 revolute joints + end fixed
    fk_result = forward_kinematics_test(test_angles)
    
    print("Joint Angles:", np.round(test_angles[1:5], 3))  # Only show the 4 active joints
    print("End-effector Position:", np.round(fk_result["position"], 3))
    print("End-effector Orientation (Quaternion):", np.round(fk_result["orientation"], 3))
    
    print("\nTesting Inverse Kinematics")
    
    # Use the position and orientation from FK as targets for IK
    target_position = fk_result["position"]
    target_orientation = fk_result["orientation"]
    
    # Run IK to find joint angles
    ik_result = inverse_kinematics_test(target_position, target_orientation, test_angles)
    
    print("Target Position:", np.round(target_position, 3))
    print("Target Orientation:", np.round(target_orientation, 3))
    print("Computed Joint Angles:", np.round(ik_result["joint_angles"][1:5], 3))  # Only show active joints
    #print("Position Error:", ik_result["position_error"])
    #print("Orientation Error:", ik_result["orientation_error"])
    
    print("\nTesting Custom Target Pose")
    
    # Define a custom target pose
    custom_position = np.array([0.06322555, -0.09916947,  0.04768746])  # Custom position
    custom_orientation = np.array([0.36301133,  0.59609231, -0.37249858,  0.61167111])  # Identity orientation (no rotation)
    
    # Run IK for the custom pose
    custom_ik_result = inverse_kinematics_test(custom_position, custom_orientation)
    
    print("Custom Target Position:", np.round(custom_position, 3))
    print("Custom Target Orientation:", np.round(custom_orientation, 3))
    print("Computed Joint Angles:", np.round(custom_ik_result["joint_angles"][1:5], 3))  # Only show active joints



   