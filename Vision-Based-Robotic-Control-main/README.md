# Vision-Based-Robotic-Control
## Chess Robot Controller with OpenMANIPULATOR-X

This project integrates a chess-playing robot arm with the OpenMANIPULATOR-X platform. The robot uses computer vision to detect chess moves and executes them using ROS-based control of the OpenMANIPULATOR-X.

## Table of Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Setup Instructions](#setup-instructions)
4. [Code Integration](#code-integration)
5. [Running the System](#running-the-system)
6. [Quick Start Guide](#quick-start-guide)

## Overview

This project allows you to:
- Detect chess moves using computer vision.
- Control the OpenMANIPULATOR-X robot arm to execute chess moves.

The project is built on **ROS Noetic** and is designed for **Ubuntu 20.04 LTS**.

## Requirements

### Hardware:
- OpenMANIPULATOR-X
- Communication Interface: U2D2 with Power Hub Board
- Power Supply: ROBOTIS SMPS 12V 5A PS-10 (recommended)

### Software:
- Ubuntu 20.04 LTS
- ROS Noetic
- Python 3.8+
- Required Python libraries: `numpy`, `opencv-python`, `pyautogui`, `scipy`, `nlohmann-json` (for C++ JSON parsing)

## Setup Instructions

### 1. Install Ubuntu
Follow the [official Ubuntu installation guide](https://ubuntu.com/tutorials/install-ubuntu-desktop) to install Ubuntu 20.04 LTS.

### 2. Install ROS Noetic
Run the following commands to install ROS Noetic:
```bash
sudo apt update
wget https://raw.githubusercontent.com/ROBOTIS-GIT/robotis_tools/master/install_ros_noetic.sh
chmod 755 ./install_ros_noetic.sh
bash ./install_ros_noetic.sh
```

### 3. Install ROS packages
```bash
source ~/.bashrc
sudo apt-get install ros-noetic-ros-controllers ros-noetic-gazebo* ros-noetic-moveit* ros-noetic-industrial-core
sudo apt install ros-noetic-dynamixel-sdk ros-noetic-dynamixel-workbench*
sudo apt install ros-noetic-robotis-manipulator
```

### 4. Download OpenManipulator-X packages
```bash
cd ~/catkin_ws/src/
git clone -b noetic https://github.com/ROBOTIS-GIT/open_manipulator.git
git clone -b noetic https://github.com/ROBOTIS-GIT/open_manipulator_msgs.git
git clone -b noetic https://github.com/ROBOTIS-GIT/open_manipulator_simulations.git
git clone -b noetic https://github.com/ROBOTIS-GIT/open_manipulator_dependencies.git
cd ~/catkin_ws && catkin_make
```

## Code Integration

### 1. Add the chessML Folder
Copy the chessML folder into your catkin_ws/src/ directory. The folder should contain:
- src with all Python scripts (chessPieceDetectorMain.py, robot_chess_controller_interpolations.py, etc.).
- CMakeLists.txt and package.xml.

### 2. Add the control_gui Folder
Copy the control_gui folder into your catkin_ws/src/open_manipulator/ directory. Ensure it contains:
- Modified qnode.cpp and qnode.hpp files.
- Other necessary files for the GUI.

### 3. Build the Workspace
Rebuild the workspace to include the new code:
```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## Running the System

Start the open manipulator x controller:
```bash
roslaunch open_manipulator_controller open_manipulator_controller.launch
```

Launch the GUI:
```bash
roslaunch open_manipulator_control_gui open_manipulator_control_gui.launch
```

Run Main script:
```bash
rosrun chessML chessPieceDetectorMain.py
```

## Quick Start Guide

### Step 1: Calibrate the Chessboard
Follow the on-screen instructions to calibrate the chessboard:
- Click the four corners of the board in the specified order.
- Press the appropriate keys to capture frames for calibration.

### Step 2: Train the Model
- Press `t` to train the chess piece detection model.

### Step 3: Play Chess
- The robot will detect your move and execute it.
- It will then calculate and execute its own move.
