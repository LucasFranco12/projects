**Projects Collection**

Welcome to my personal Projects Collection! This repository serves as a central hub for showcasing various projects I've worked on, spanning domains like machine learning, robotics, Rust development, and web tools. Browse the list below to explore each project.

---

## Table of Contents

* [ChessML](#chessml)
* [FitTrack](#fittrack)
* [PYTHONAI](#pythonai)
* [SmartGit](#smartgit)
* [Vision-Based Robotic Control](#vision-based-robotic-control)
* [Battleship](#battleship)
* [chessFun](#chessfun)
* [playbutton](#playbutton)
* [Sustainability Database](#sustainability-database)
* [Website](#website)
* [YouTube Transcript AI](#youtube-transcript-ai)

---

### ChessML

**Path:** `ChessML-main/`
A scratchpad for miscellaneous machine learning and robotics scripts used during development of my chess-playing robot. Contains experiments and utilities for piece detection and arm control, including:

boundingBoxPrediction/: Scripts to generate and visualize bounding box predictions on chessboard images.

chessML/: Assorted ML experiments for chess analysis.

chessPieceDetectorYOLOmodelexDATA/: YOLO model files and training data for piece detection.

csv_annotations/: Conversion scripts for centralized annotation JSON to CSV format.

IKFK.py: Inverse kinematics and forward kinematics utilities to assist calibration of the robotic arm.

qnode.cpp / qnode.hpp: ROS Qt node source for interfacing vision predictions with the robot arm via ROS.

chessPieceDetector.py / chessPieceDetectorENGINETEST.py: Python scripts for testing detection pipelines.
### FitTrack

**Path:** `FitTrack-main/`
A comprehensive fitness and nutrition tracking mobile app built with React Native (Expo) that empowers users to log workouts, scan food barcodes/QR codes, and analyze detailed nutritional data. Key components include:

Barcode & QR Scanning (android/): Capture product barcodes or QR codes to fetch nutrition information from the FDA food database in real time.

Nutrition Algorithms (ai/): Custom Python scripts and algorithms to calculate macronutrient/micronutrient breakdowns and daily targets based on user profiles.

Exercise Logging (components/, screens/): Record workouts, sets, reps, and weights through intuitive UI screens and reusable components.

Progress Visualization (utils/): Interactive charts and metrics to track weight, body measurements, and nutrient intake over time.

AI Personal Trainer (ai/): Experimental AI module offering meal plans and workout suggestions, informed by a curated library of 200+ health and science research papers.

Modular Architecture: Well-structured folders for assets/, navigation/, constants/, and utils/ to facilitate scalable feature development and easy maintenance.

Refer to the in-project README for installation, Expo setup, and usage instructions

### PYTHONAI

**Path:** `PYTHONAI/`
An experimental collection of standalone Python AI and machine learning scripts demonstrating fundamental algorithms and practical demos. Highlights include:

AmazonCommentorBias.py: Exploratory analysis of bias in Amazon product reviews.

BodyWeightLinearRegression.py: Linear regression model to predict body weight from anthropometric measurements.

blackjackAI.py: Reinforcement learning agent for training and evaluating strategies in the game of Blackjack.

decisionTree.py: Implementation of decision tree classification from scratch, including data preprocessing and visualization.

kmeans.py: K-means clustering algorithm applied to sample datasets for unsupervised learning demonstrations.

knn.py: k-Nearest Neighbors classifier for classification tasks with configurable distance metrics.

linearRegression.py: Basic linear regression example using NumPy for educational purposes.

logisticRegression.py: Logistic regression classifier to predict binary outcomes with training and evaluation scripts.

neuralNetworkHyperparameter.py: Pipeline for tuning hyperparameters of a simple feedforward neural network using grid search.

Each script is self-contained and showcases one core ML concept. Feel free to run experiments, modify parameters, and extend the demos for your own learning.### SmartGit
### SmartGit

**Path:** `SmartGit-main/`
A custom Git client implemented in Rust. Features include:

* Background auto-commit daemon (`smartgit watch`)
* Colored CLI output and progress bars
* Metadata syncing with Firestore
* Custom diff and file-history commands
* AI COMMENTED COMMITS / AI generated Tasks to accomplish your repo's goals
* Makes Git a fun game... still work in progress

### Vision-Based Robotic Control

**Path:** `Vision-Based-Robotic-Control-main/`
A full ROS-based chess robot framework that uses computer vision to interpret human moves and commands an OpenMANIPULATOR-X arm to execute them on a physical board. Core components include:

chessML/: Vision pipeline scripts for board calibration, image preprocessing, and piece detection using OpenCV and YOLO/SVM models, along with move inference logic.

open_manipulator_control_gui/: Qt-based ROS node (qnode.cpp/hpp) providing a graphical interface for monitoring robot status, manual overrides, and calibration workflows.

Launch Configurations: Ready-to-use ROS launch files for the manipulator controller, vision nodes, GUI, and simulation environments.

Calibration Tools: Interactive routines to locate chessboard corners and map pixel coordinates to manipulator coordinates for precise motion planning.

Control Loop: Python and C++ nodes that implement the end-to-end loop: detect player move → compute robot trajectory → execute via OpenMANIPULATOR-X.

Requirements: Ubuntu 20.04 LTS, ROS Noetic, Python 3.8+, ROBOTIS U2D2 interface, and a 12V power supply (SMPS PS-10 recommended).Refer to the project’s own README for detailed setup, hardware connections, and usage instructions.### Battleship

### Battleship

**Path:** `battleship-LucasFranco12-main/`
A networked multiplayer Battleship game implemented in C++ that supports up to 30 concurrent players. Core components include:

Makefile: Build configuration for compiling the server and client executables.

network_packet.h: Definitions for custom TCP packet structures used for game messaging (ship placement, attack commands, status updates).

server.cpp: Central server handling client connections, game lobbies, turn management, and message routing to synchronize game state.

test_client.cpp: Reference client demonstrating how to connect to the server, send commands (place ships, fire shots), and render console-based board updates.

Gameplay Flow:

Clients connect to the server and are assigned player IDs.

Each player places ships on their grid.

Players take turns sending attack coordinates; server validates hits/misses and broadcasts results.

Game continues until one player's ships are all sunk.

Refer to the in-project README for details on configuring ports and running multiple clients.Licensed under MIT.

### chessFun

**Path:** `chessFun/`
An experimental pipeline of OpenCV and machine learning scripts designed to detect and track chess pieces from a live camera feed, analyze board states, and predict player moves in real time. Key components include:

templates/: Sample board images and template assets for calibration, template matching, and testing.

cameraVisionTest.py: Initial OpenCV-based script for webcam capture, board detection, and square segmentation.

cameraVisionV2.py to cameraVisionV6.py: Iterative enhancements implementing feature extraction, adaptive thresholding, contour analysis, and custom heuristics to improve piece detection accuracy.

cameraVisionNeuralNetworkTest.py: Neural network classifier prototype for identifying chess pieces using a trained model.

chessArena.py: Integration script that consolidates vision outputs, reconstructs the board state, and interfaces with a chess engine to suggest or validate moves.

Each version explores distinct image processing and ML techniques to refine performance. Feel free to adjust parameters, swap models, or extend the pipeline for your own experiments.

### playbutton

**Path:** `playbutton/`
A small web component showcasing a customizable "Play" button UI element built with HTML, CSS, and vanilla JavaScript. Demonstrates interactive hover and click animations.

### Sustainability Database

**Path:** `sustainability-database-main/`
A prototype database-driven web application for exploring environmental sustainability indicators at county and municipal levels. Key components include:

DataFiles/: Raw CSV datasets of sustainability metrics sourced from public databases and research reports.

database/: SQL scripts defining the schema and the format_table.sh shell script to create and populate the local SQLite database.

src/: Flask application (app.py) that exposes routes for querying and visualizing data, and provides the web UI logic.

images/: Screenshot assets demonstrating the user interface for county vs. municipality data views.

Usage:

Create and populate the database:

cd database
sh format_table.sh

Launch the web application:

export FLASK_APP=src/app.py
flask run

Open your browser at http://127.0.0.1:5000/, choose between county or municipality metrics, and submit to view interactive tables and charts.

Refer to the in-project README for environment setup, dependencies (Flask, SQLite3), and customization tips.### Website
### Website

**Path:** `website/`
My personal portfolio website source code. Built with HTML, CSS, and JavaScript, featuring sections for projects, resume, and contact information.

### YouTube Transcript AI

**Path:** `youtubeTranscriptAI/`
A Python-driven workflow that automates the extraction of YouTube video transcripts and generates structured quiz outlines using a free LLM API. Core components include:

transcriptGrabber.py: Downloads and parses transcripts from YouTube videos via the YouTube Data API or caption scraping.

ai.py: Encapsulates LLM API calls, handling authentication with your API key and applying a customizable quiz-outline prompt template.

combinedAPP.py: End-to-end script that ties together transcript retrieval and LLM processing to output a quiz structure (questions, options, and answers).

app/: Optional Flask-based interface hosting index.html, where users can input a video URL and receive a generated quiz outline in the browser.

templates/: Jinja2 templates for rendering quiz outlines in HTML format.

New Text Document.txt: Example prompt and sample output illustrating how transcripts map to quiz content.

Usage:

Set your LLM API key as an environment variable (e.g., export LLM_API_KEY=your_key).

Run the combined script:

python combinedAPP.py --video-url <YouTube_URL>

Review the generated quiz outline in the console or via the Flask web UI at http://127.0.0.1:5000/.


---

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/LucasFranco12/projects.git
   cd projects
   ```

2. **Navigate to a project folder** and follow the individual README or instructions within that directory.

3. **Dependencies** vary by project. Most Python-based projects include a `requirements.txt` or `environment.yml`.

---

## Contributing

Feel free to submit issues or pull requests! For guidelines, refer to the [CONTRIBUTING.md](#) (coming soon).

---

## License

This collection is distributed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Made with ❤️ by Lucas Franco
