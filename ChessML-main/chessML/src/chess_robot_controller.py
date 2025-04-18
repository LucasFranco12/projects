#!/usr/bin/env python3
#lets create a simple python script that can print out the joint angles with a press of a key
#lets also print out the x,y,z coordinates of the end effector
#then with another press of a key send it to the home position through joint angles
#then with another press of a key send it to the home position through x,y,z coordinates
#then another key press to move the robot to a specific position through joint angles
#another key press to move the robot to a specific position through x,y,z coordinates
#another key press to move the robot to a specific position through a trajectory

import rospy
import numpy as np
import termios
import sys
import tty
import select
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped
from open_manipulator_msgs.msg import OpenManipulatorState, KinematicsPose

# Global variables to store current state
current_state = None
current_kinematics = None

def states_callback(msg):
    """Callback for robot state"""
    global current_state
    current_state = msg
    
def kinematics_callback(msg):
    """Callback for end effector pose"""
    global current_kinematics
    current_kinematics = msg

def get_key():
    """Get a single keypress from the user"""
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def print_robot_state():
    """Print current robot state"""
    if current_state is None:
        print("Robot state not received yet.")
        return
    
    print(f"\nCurrent Robot State: {current_state.open_manipulator_moving_state}")

def print_end_effector_position():
    """Print current end effector position"""
    if current_kinematics is None:
        print("End effector pose not received yet.")
        return
    
    pose = current_kinematics.pose
    print("\nCurrent End Effector Position:")
    print(f"  Position: x={pose.position.x:.4f}, y={pose.position.y:.4f}, z={pose.position.z:.4f}")
    print(f"  Orientation: x={pose.orientation.x:.4f}, y={pose.orientation.y:.4f}, " +
          f"z={pose.orientation.z:.4f}, w={pose.orientation.w:.4f}")

def move_to_home_joint():
    """Move to home position using joint angles"""
    # Define home position joint angles
    home_position = [0.0, 0.0, 0.0, 0.0, 0.0]  # Adjust based on your robot
    
    # Publish to each joint
    joint1_pub.publish(Float64(home_position[0]))
    joint2_pub.publish(Float64(home_position[1]))
    joint3_pub.publish(Float64(home_position[2]))
    joint4_pub.publish(Float64(home_position[3]))
    gripper_pub.publish(Float64(home_position[4]))
    
    print("Moving to home position using joint angles.")

def move_to_custom_joint():
    """Move to a custom position using joint angles"""
    # Define a custom position
    custom_joint_position = [0.5, -0.3, 0.2, 0.8, 0.0]  # Adjust for your robot
    
    # Publish to each joint
    joint1_pub.publish(Float64(custom_joint_position[0]))
    joint2_pub.publish(Float64(custom_joint_position[1]))
    joint3_pub.publish(Float64(custom_joint_position[2]))
    joint4_pub.publish(Float64(custom_joint_position[3]))
    gripper_pub.publish(Float64(custom_joint_position[4]))
    
    print("Moving to custom position using joint angles.")

def execute_trajectory():
    """Execute a simple trajectory one point at a time"""
    # Define a sequence of points
    positions = [
        [0.0, 0.0, 0.0, 0.0, 0.0],  # Start position
        [0.3, -0.2, 0.1, 0.3, 0.0],  # Waypoint 1
        [0.5, -0.3, 0.2, 0.5, 0.0],  # Waypoint 2
        [0.7, -0.2, 0.1, 0.3, 0.0],  # Waypoint 3
        [0.0, 0.0, 0.0, 0.0, 0.0]   # End position
    ]
    
    print("Executing trajectory...")
    for i, pos in enumerate(positions):
        print(f"Moving to waypoint {i+1}/{len(positions)}")
        joint1_pub.publish(Float64(pos[0]))
        joint2_pub.publish(Float64(pos[1]))
        joint3_pub.publish(Float64(pos[2]))
        joint4_pub.publish(Float64(pos[3]))
        gripper_pub.publish(Float64(pos[4]))
        rospy.sleep(2.0)  # Wait for movement to complete
    
    print("Trajectory completed.")

def input_custom_angles():
    """Allow user to input custom joint angles"""
    print("\nEnter joint angles in radians:")
    try:
        j1 = float(input("Joint 1: "))
        j2 = float(input("Joint 2: "))
        j3 = float(input("Joint 3: "))
        j4 = float(input("Joint 4: "))
        gripper = float(input("Gripper: "))
        
        # Publish to each joint
        joint1_pub.publish(Float64(j1))
        joint2_pub.publish(Float64(j2))
        joint3_pub.publish(Float64(j3))
        joint4_pub.publish(Float64(j4))
        gripper_pub.publish(Float64(gripper))
        
        print(f"Moving to position: [{j1}, {j2}, {j3}, {j4}, {gripper}]")
    except ValueError:
        print("Invalid input. Please enter valid numbers.")

def print_menu():
    """Print the control menu"""
    print("\n=== Chess Robot Controller ===")
    print("Press a key to perform an action:")
    print("  1 - Print robot state")
    print("  2 - Print current end effector position")
    print("  3 - Move to home position (joint angles)")
    print("  4 - Move to custom position (joint angles)")
    print("  5 - Execute simple trajectory")
    print("  6 - Input custom joint angles")
    print("  q - Quit")
    print("================================")

def main():
    global settings, joint1_pub, joint2_pub, joint3_pub, joint4_pub, gripper_pub
    
    # Initialize ROS node
    rospy.init_node('chess_robot_controller', anonymous=True)
    
    # Save terminal settings for key input
    settings = termios.tcgetattr(sys.stdin)
    
    # Set up subscribers
    rospy.Subscriber("/states", OpenManipulatorState, states_callback)
    rospy.Subscriber("/gripper/kinematics_pose", KinematicsPose, kinematics_callback)
    
    # Set up publishers for individual joints
    joint1_pub = rospy.Publisher("/joint1_position/command", Float64, queue_size=1)
    joint2_pub = rospy.Publisher("/joint2_position/command", Float64, queue_size=1)
    joint3_pub = rospy.Publisher("/joint3_position/command", Float64, queue_size=1)
    joint4_pub = rospy.Publisher("/joint4_position/command", Float64, queue_size=1)
    gripper_pub = rospy.Publisher("/gripper_position/command", Float64, queue_size=1)
    
    # Wait for publishers to be ready
    rospy.sleep(1.0)
    
    print("Chess Robot Controller started. Press any key to display the menu.")
    
    try:
        while not rospy.is_shutdown():
            print_menu()
            key = get_key()
            
            if key == '1':
                print_robot_state()
            elif key == '2':
                print_end_effector_position()
            elif key == '3':
                move_to_home_joint()
            elif key == '4':
                move_to_custom_joint()
            elif key == '5':
                execute_trajectory()
            elif key == '6':
                input_custom_angles()
            elif key == 'q':
                break
            
            rospy.sleep(0.1)
            
    except Exception as e:
        print(e)
        
    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        print("Exiting Chess Robot Controller")

if __name__ == '__main__':
    main()