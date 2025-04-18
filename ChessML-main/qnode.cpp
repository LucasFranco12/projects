/*******************************************************************************
* Copyright 2018 ROBOTIS CO., LTD.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/* Authors: Darby Lim, Hye-Jong KIM, Ryan Shim, Yong-Ho Na */

/*****************************************************************************
** Includes
*****************************************************************************/

#include <ros/ros.h>
#include <ros/network.h>
#include <string>
#include <std_msgs/String.h>
#include <sstream>
#include <thread>
#include <chrono>
#include "../include/open_manipulator_control_gui/qnode.hpp"
#include <nlohmann/json.hpp>

/*****************************************************************************
** Namespaces
*****************************************************************************/

namespace open_manipulator_control_gui {

/*****************************************************************************
** Implementation
*****************************************************************************/

QNode::QNode(int argc, char** argv ) :
	init_argc(argc),
  init_argv(argv),
  open_manipulator_actuator_enabled_(false),
  open_manipulator_is_moving_(false)
	{
    
  }


QNode::~QNode() {
    if(ros::isStarted()) {
      ros::shutdown(); // explicitly needed since we use ros::start();
      ros::waitForShutdown();
    }
	wait();
}

bool QNode::init() {
  ros::init(init_argc, init_argv, "open_manipulator_control_gui");
  if (!ros::master::check()) {
    return false;
  }
  ros::start();
  ros::NodeHandle n;

  // Initialize subscribers
  chess_move_sub_ = n.subscribe("/chess_moves", 10, &QNode::chessMoveCallback, this);

  // msg publisher
  open_manipulator_option_pub_ = n.advertise<std_msgs::String>("option", 10);
  // msg subscriber
  open_manipulator_states_sub_       = n.subscribe("states", 10, &QNode::manipulatorStatesCallback, this);
  open_manipulator_joint_states_sub_ = n.subscribe("joint_states", 10, &QNode::jointStatesCallback, this);
  open_manipulator_kinematics_pose_sub_ = n.subscribe("/open_manipulator/kinematics_pose", 10, &QNode::kinematicsPoseCallback, this);
  ROS_INFO("Subscribed to /open_manipulator/kinematics_pose");
  // service client
  goal_joint_space_path_client_ = n.serviceClient<open_manipulator_msgs::SetJointPosition>("goal_joint_space_path");
  goal_task_space_path_position_only_client_ = n.serviceClient<open_manipulator_msgs::SetKinematicsPose>("goal_task_space_path_position_only");
  goal_tool_control_client_ = n.serviceClient<open_manipulator_msgs::SetJointPosition>("goal_tool_control");
  set_actuator_state_client_ = n.serviceClient<open_manipulator_msgs::SetActuatorState>("set_actuator_state");
  goal_drawing_trajectory_client_ = n.serviceClient<open_manipulator_msgs::SetDrawingTrajectory>("goal_drawing_trajectory");
  present_kinematic_position_.resize(3, 0.0);

  start();
  return true;
}

void QNode::run() {
  ros::Rate loop_rate(10);
	while ( ros::ok() ) {
		ros::spinOnce();
		loop_rate.sleep();
	}
	std::cout << "Ros shutdown, proceeding to close the gui." << std::endl;
	Q_EMIT rosShutdown();
}

void QNode::manipulatorStatesCallback(const open_manipulator_msgs::OpenManipulatorState::ConstPtr &msg)
{
  if(msg->open_manipulator_moving_state == msg->IS_MOVING)
    open_manipulator_is_moving_ = true;
  else
    open_manipulator_is_moving_ = false;

  if(msg->open_manipulator_actuator_state == msg->ACTUATOR_ENABLED)
    open_manipulator_actuator_enabled_ = true;
  else
    open_manipulator_actuator_enabled_ = false;
}
void QNode::jointStatesCallback(const sensor_msgs::JointState::ConstPtr &msg)
{
  std::vector<double> temp_angle;
  temp_angle.resize(NUM_OF_JOINT_AND_TOOL);
  for(int i = 0; i < msg->name.size(); i ++)
  {
    if(!msg->name.at(i).compare("joint1"))  temp_angle.at(0) = (msg->position.at(i));
    else if(!msg->name.at(i).compare("joint2"))  temp_angle.at(1) = (msg->position.at(i));
    else if(!msg->name.at(i).compare("joint3"))  temp_angle.at(2) = (msg->position.at(i));
    else if(!msg->name.at(i).compare("joint4"))  temp_angle.at(3) = (msg->position.at(i));
    else if(!msg->name.at(i).compare("gripper"))  temp_angle.at(4) = (msg->position.at(i));
  }
  present_joint_angle_ = temp_angle;
}

void QNode::kinematicsPoseCallback(const open_manipulator_msgs::KinematicsPose::ConstPtr &msg)
{
  try {
    present_kinematic_position_[0] = msg->pose.position.x;
    present_kinematic_position_[1] = msg->pose.position.y;
    present_kinematic_position_[2] = msg->pose.position.z;

    ROS_DEBUG("Updated kinematic pose: x=%f, y=%f, z=%f", 
             present_kinematic_position_[0],
             present_kinematic_position_[1],
             present_kinematic_position_[2]);
  } catch (const std::exception& e) {
    ROS_ERROR("Error in kinematicsPoseCallback: %s", e.what());
  }
}

std::vector<double> QNode::getPresentJointAngle()
{
  return present_joint_angle_;
}
std::vector<double> QNode::getPresentKinematicsPose()
{
  return present_kinematic_position_;
}
bool QNode::getOpenManipulatorMovingState()
{
  return open_manipulator_is_moving_;
}
bool QNode::getOpenManipulatorActuatorState()
{
  return open_manipulator_actuator_enabled_;
}

void QNode::setOption(std::string opt)
{
  std_msgs::String msg;
  msg.data = opt;
  open_manipulator_option_pub_.publish(msg);
}

bool QNode::setJointSpacePath(std::vector<std::string> joint_name, std::vector<double> joint_angle, double path_time)
{
  open_manipulator_msgs::SetJointPosition srv;
  srv.request.joint_position.joint_name = joint_name;
  srv.request.joint_position.position = joint_angle;
  srv.request.path_time = path_time;

  if(goal_joint_space_path_client_.call(srv))
  {
    return srv.response.is_planned;
  }
  return false;
}

bool QNode::setTaskSpacePath(std::vector<double> kinematics_pose, double path_time)
{
    if (kinematics_pose.size() < 3) {
        ROS_ERROR("Kinematics pose vector must contain at least 3 elements (x, y, z)");
        return false;
    }

    open_manipulator_msgs::SetKinematicsPose srv;
    srv.request.end_effector_name = "gripper";

    srv.request.kinematics_pose.pose.position.x = kinematics_pose[0];
    srv.request.kinematics_pose.pose.position.y = kinematics_pose[1];
    srv.request.kinematics_pose.pose.position.z = kinematics_pose[2];

// Keep orientation constant (facing downward)
    srv.request.kinematics_pose.pose.orientation.w = 1.0;
    srv.request.kinematics_pose.pose.orientation.x = 0.0;
    srv.request.kinematics_pose.pose.orientation.y = 0.0;
    srv.request.kinematics_pose.pose.orientation.z = 0.0;

    srv.request.path_time = path_time;

    ROS_INFO("Sending task space path request - Position: [%f, %f, %f], Time: %f", 
             kinematics_pose[0], kinematics_pose[1], kinematics_pose[2], path_time);

    if (goal_task_space_path_position_only_client_.call(srv)) {
        if (srv.response.is_planned) {
            ROS_INFO("Task space path planned successfully");
            return true;
        } else {
            ROS_ERROR("Task space path planning failed");
            return false;
        }
    }

    ROS_ERROR("Failed to call task space path service");
    return false;
}

bool QNode::setDrawingTrajectory(std::string name, std::vector<double> arg, double path_time)
{
  open_manipulator_msgs::SetDrawingTrajectory srv;
  srv.request.end_effector_name = "gripper";
  srv.request.drawing_trajectory_name = name;
  srv.request.path_time = path_time;
  for(int i = 0; i < arg.size(); i ++)
    srv.request.param.push_back(arg.at(i));

  if(goal_drawing_trajectory_client_.call(srv))
  {
    return srv.response.is_planned; 
  }
  return false;
}

bool QNode::setToolControl(std::vector<double> joint_angle)
{
  open_manipulator_msgs::SetJointPosition srv;
  srv.request.joint_position.joint_name.push_back("gripper");
  srv.request.joint_position.position = joint_angle;

  if(goal_tool_control_client_.call(srv))
  {
    return srv.response.is_planned;
  }
  return false;
}

bool QNode::setActuatorState(bool actuator_state)
{
  open_manipulator_msgs::SetActuatorState srv;
  srv.request.set_actuator_state = actuator_state;

  if(set_actuator_state_client_.call(srv))
  {
    return srv.response.is_planned;
  }
  return false;
}

void QNode::chessMoveCallback(const std_msgs::String::ConstPtr &msg)
{
    std::string trajectory_json = msg->data;
    log(Info, "Received chess move trajectory: " + trajectory_json);  // Log the received trajectory

    // Parse and execute the trajectory
    executeJointTrajectory(trajectory_json);
}

void QNode::executeJointTrajectory(const std::string &trajectory_json)
{
    // Parse the JSON string containing the joint trajectory
    try {
        // Use the JSON library to parse the string
        auto json_doc = nlohmann::json::parse(trajectory_json);
        
        // Should be an array of arrays
        if (!json_doc.is_array()) {
            log(Error, "Invalid trajectory format: not an array");
            return;
        }
        
        int num_points = json_doc.size();
        log(Info, "Executing trajectory with " + std::to_string(num_points) + " points");
        
        // Open the gripper slightly before starting any movement
        if (setToolControl({0.01})) {  // Open gripper
            log(Info, "Opened gripper to start movement");
            std::this_thread::sleep_for(std::chrono::seconds(1));
        } else {
            log(Error, "Failed to open gripper before starting");
            return;
        }
        
        // Execute each trajectory point
        for (int i = 0; i < num_points; i++) {
            auto point = json_doc[i];
            if (!point.is_array() || point.size() < 6) {
                log(Error, "Invalid point format at index " + std::to_string(i));
                continue;
            }
            
            // Extract joint angles from the point
            std::vector<double> joint_angles;
            for (int j = 0; j < 4; j++) {
                // Use joint1 to joint4 
                joint_angles.push_back(point[j+1].get<double>());
            }
            
            // Joint names for the service call
            std::vector<std::string> joint_names = {"joint1", "joint2", "joint3", "joint4"};
            
            // Log the joint angles
            std::stringstream ss;
            ss << "Moving to trajectory point " << (i+1) << ": [";
            for (size_t j = 0; j < joint_angles.size(); j++) {
                ss << joint_angles[j];
                if (j < joint_angles.size() - 1) ss << ", ";
            }
            ss << "]";
            log(Info, ss.str());
            
            // Move to the joint position
            if (setJointSpacePath(joint_names, joint_angles, 2.0)) {
                log(Info, "Reached trajectory point " + std::to_string(i+1));
                
                // Wait for movement to complete
                std::this_thread::sleep_for(std::chrono::seconds(2));
                
                // Control gripper based on trajectory point
                if (num_points > 6) {  // This is a capture move
                    // For capture moves
                    if (i == 1) {  // Close gripper to grab captured piece
                        setToolControl({0});
                    } else if (i == 4) {  // Open gripper to drop captured piece
                        setToolControl({0.01});
                    } else if (i == 6) {  // Close gripper to grab moving piece
                        setToolControl({0});
                    } else if (i == 9) {   // Open gripper to place moving piece
                        setToolControl({0.01});
                    }
                } else {  // Standard move
                    // Existing logic for standard moves
                    if (i == 1) {
                        setToolControl({0});  // Close gripper
                    } else if (i == 4) {
                        setToolControl({0.01});  // Open gripper
                    }
                }
            } else {
                log(Error, "Failed to reach trajectory point " + std::to_string(i+1));
        return;
    }
        }
        // Move to final resting position
        std::vector<std::string> joint_names = {"joint1", "joint2", "joint3", "joint4"};
        std::vector<double> final_angles = {-0.092, -0.906, -0.569, 0.911};
        
        log(Info, "Moving to final resting position");
        if (setJointSpacePath(joint_names, final_angles, 2.0)) {
            log(Info, "Reached final resting position");
            std::this_thread::sleep_for(std::chrono::seconds(2));
        } else {
            log(Error, "Failed to reach final resting position");
            return;
        }
        log(Info, "Joint trajectory execution completed successfully");
        
    } catch (const std::exception& e) {
        log(Error, "Error parsing or executing trajectory: " + std::string(e.what()));
    }
}



void QNode::log(const LogLevel &level, const std::string &msg) {
    logging_model.insertRows(logging_model.rowCount(), 1);
    std::stringstream logging_model_msg;
    switch (level) {
        case Debug:
            ROS_DEBUG_STREAM(msg);
            logging_model_msg << "[DEBUG] " << msg;
            break;
        case Info:
            ROS_INFO_STREAM(msg);
            logging_model_msg << "[INFO] " << msg;
            break;
        case Warn:
            ROS_WARN_STREAM(msg);
            logging_model_msg << "[WARN] " << msg;
            break;
        case Error:
            ROS_ERROR_STREAM(msg);
            logging_model_msg << "[ERROR] " << msg;
            break;
        case Fatal:
            ROS_FATAL_STREAM(msg);
            logging_model_msg << "[FATAL] " << msg;
            break;
    }
    QVariant new_row(QString(logging_model_msg.str().c_str()));
    logging_model.setData(logging_model.index(logging_model.rowCount() - 1), new_row);
    Q_EMIT loggingModel();  // Emit a signal if you want to update the GUI log
}

}  // namespace open_manipulator_control_gui
