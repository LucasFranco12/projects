#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64
import time
from chessPieceDetectorENGINETEST import RobotChessController

def main():
    # Initialize ROS node
    rospy.init_node('robot_arm_test', anonymous=True)
    
    # Create robot controller
    robot = RobotChessController()
    
    # Wait for the controller to initialize
    while not robot.is_initialized:
        print("Waiting for robot controller to initialize...")
        time.sleep(1)
    
    print("Robot controller initialized successfully!")
    
    # Test moving to some predefined positions
    test_positions = [
        # Move to a8 (top-left corner)
        {'square': 'a8', 'description': 'Moving to top-left corner (a8)'},
        # Move to h8 (top-right corner)
        {'square': 'h8', 'description': 'Moving to top-right corner (h8)'},
        # Move to h1 (bottom-right corner)
        {'square': 'h1', 'description': 'Moving to bottom-right corner (h1)'},
        # Move to a1 (bottom-left corner)
        {'square': 'a1', 'description': 'Moving to bottom-left corner (a1)'}
    ]
    
    for position in test_positions:
        print(f"\n{position['description']}")
        
        # Get the joint angles for this position
        if position['square'] in robot.calibrated_positions:
            joint_angles = robot.calibrated_positions[position['square']]
            
            # Publish joint angles to the robot
            robot.joint1_pub.publish(joint_angles[1])
            robot.joint2_pub.publish(joint_angles[2])
            robot.joint3_pub.publish(joint_angles[3])
            robot.joint4_pub.publish(joint_angles[4])
            robot.gripper_pub.publish(joint_angles[5])
            
            print(f"Published joint angles: {joint_angles}")
            time.sleep(2)  # Wait for the arm to move
        else:
            print(f"Position {position['square']} not found in calibrated positions")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass


