[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/5mCoF9-h)
# TOBB ETÜ ELE495 - Capstone Project

# Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Acknowledgements](#acknowledgements)

## Introduction
Currently, autonomous vehicle technologies are rapidly developing and finding applications in various fields. Autonomous parking systems are of great importance, especially for solving the problem of finding parking spaces in crowded cities. In this project, an autonomous vehicle parking system will be designed using Jetson Nano and JetBot. The goal is to design a parking system with a vehicle that can autonomously park in a designated parking area through a mobile application.

## Features
Key features and functionalities of the project.
### Hardware:
- JetBot               : A small, affordable AI robot built around the NVIDIA Jetson Nano.
- NVIDIA Jetson Nano   : A small, powerful computer for AI development.
- CSI Camera Module    : Used for capturing video feed for digit recognition.
- TCS3200 Color Sensor : Used to detect the color red.
- Wi-Fi Module         : Provides wireless connectivity for remote control and communication.


### Operating System and packages
The project runs on the NVIDIA Jetson Nano, which uses the Ubuntu-based JetPack operating system. The key packages and libraries used include:
- JetPack SDK : Provides a full development environment for AI applications.
- Python      : The programming language used for the project scripts. The following Python libraries and packages are utilized in the project;
  - **cv2 (OpenCV)**: Used for video capture and image processing.
  - **time**: Used for handling timing operations.
  - **math**: Provides mathematical functions.
  - **jetbot**: Specific library for controlling the JetBot.
  - **os**: Used for interacting with the operating system.
  - **numpy**: A fundamental package for scientific computing with Python.
  - **torch**: An open-source machine learning library used for developing the digit recognition model.
  - **torch.nn**: Used for constructing neural networks.
  - **torch.nn.functional**: Provides additional functionalities for neural networks.
  - **torchvision.transforms**: Used for image transformations.
  - **PIL (Pillow)**: Used for image processing.
  - **requests**: Allows sending HTTP requests.
- PyTorch: PyTorch is an open-source machine learning library used for developing the digit recognition model in this project. The project utilizes PyTorch for several key tasks, including loading a pre-trained model, transforming images, and running inference. Here’s a detailed explanation of these tasks:

  - **Model Loading**: The project uses PyTorch to load a pre-trained model. This is done using the `torch.jit.load` function, which allows the model to be loaded in a serialized format, making it suitable for production deployment. Once loaded, the model is set to evaluation mode with `model.eval()`. This is important as it disables certain layers like dropout and batch normalization, which are only needed during training.

  - **Image Transformations**: Before feeding images into the model, they must be preprocessed and transformed. This involves several steps:
    - **Grayscale Conversion**: Images are converted to grayscale to reduce complexity and focus on essential features.
    - **Resizing**: Images are resized to a fixed size that the model expects.
    - **Tensor Conversion**: Images are converted into PyTorch tensors, the format required for model input.
    - **Normalization**: Image data is normalized to ensure it has a consistent mean and standard deviation, which helps in achieving better model performance.

  - **Image Preprocessing Function**: A specific function is used to preprocess the images captured from the camera. This function handles converting the image to grayscale, applying a binary threshold to highlight key features, converting it to a suitable format, and then applying the predefined transformations. This step is crucial for ensuring that the images fed into the model are in the correct format and contain the necessary information for accurate predictions.

  - **Model Inference**: During inference, the preprocessed image tensor is fed into the model to obtain predictions. The inference process is run within a `torch.no_grad()` context to disable gradient calculations, which are unnecessary for inference and help in reducing computational overhead and memory usage. The model outputs probabilities for each class (digit), from which the most probable class is selected.

  - **Integration with JetBot**: The predictions from the model are integrated into the JetBot's control logic. Based on the recognized digits, the JetBot makes real-time decisions for navigation and parking. This integration demonstrates how PyTorch models can be effectively used in robotic applications to enhance autonomous functionalities.

- Overall, PyTorch provides a robust and flexible framework for developing and deploying machine learning models, and its use in this project showcases its capabilities in a real-world application involving image recognition and robotic control.

- GStreamer   : GStreamer is a pipeline-based multimedia framework used to handle video streams. In this project, GStreamer is utilized to capture video input from a camera and process it in real-time.
### Applications 
1. **Digit Recognition**:
    - **Purpose**: The primary application of this project is to recognize digits from a live camera feed.
    - **Implementation**:
        - The JetBot captures video frames using a camera module.
        - Each frame is preprocessed and fed into a pre-trained PyTorch model.
        - The model predicts the digit in the frame, which can be used for various decision-making tasks.
    - **Usage**: This application can be used in scenarios where real-time digit recognition is needed, such as reading meter readings, recognizing numbers on signs, or any other application requiring digit identification.

2. **Color Detection**:
    - **Purpose**: To detect specific colors in the environment, specifically the color red.
    - **Implementation**:
        - The TCS3200 color sensor is used to detect the color red.
        - The sensor data is processed to determine the presence and intensity of the color.
    - **Usage**: This application can be used in scenarios such as line following robots, sorting systems in factories, or any other application where color detection is required.

3. **Real-time Video Processing**:
    - **Purpose**: To handle and process live video streams for real-time applications.
    - **Implementation**:
        - GStreamer is used to set up a pipeline for capturing video input from the camera.
        - The video frames are then processed using OpenCV for image processing tasks.
    - **Usage**: This enables the JetBot to process and analyze video in real-time, making it suitable for applications like surveillance, navigation, and object tracking.

4. **Remote Monitoring and Control**:
    - **Purpose**: To allow remote control and monitoring of the JetBot over a network.
    - **Implementation**:
        - The WiFi module provides wireless connectivity.
        - Flask-based web services handle communication between the JetBot and remote clients.
    - **Usage**: This application can be used for remote inspection, teleoperation, and data collection, allowing users to control and monitor the JetBot from a distance.

5. **Autonomous Navigation**:
    - **Purpose**: To enable the JetBot to navigate autonomously using visual cues and sensor data.
    - **Implementation**:
        - The JetBot uses the digit recognition model and color detection to make navigation decisions.
        - A PID controller is implemented for precise motor control based on sensor inputs.
    - **Usage**: This allows the JetBot to move through environments without human intervention, making it useful for applications in exploration, search and rescue, and automated delivery.
    - 
6. **Mobile Application**:
    - **Purpose**: To provide a user-friendly interface for controlling and monitoring the JetBot.
    - **Implementation**:
        - Developed a mobile application that connects to the JetBot via the WiFi module.
        - The app allows users to view the live video feed, send control commands, and monitor sensor data.
    - **Usage**: Enhances the usability of the JetBot by providing an intuitive interface for remote operation and real-time monitoring.

### Services

1. **Real-time Video Processing**:
    - **Description**: The project captures and processes live video streams from the camera in real-time.
    - **Implementation**:
        - Utilizes GStreamer to set up a pipeline for capturing video input.
        - Processes video frames using OpenCV for image preprocessing and transformation.
    - **Benefit**: Enables the JetBot to analyze and react to visual input instantly, which is crucial for tasks such as digit recognition and navigation.

2. **Model Deployment**:
    - **Description**: The project deploys a pre-trained PyTorch model for real-time digit recognition.
    - **Implementation**:
        - Loads the model using PyTorch's `torch.jit.load` for efficient inference.
        - Runs inference on preprocessed image frames to predict digits.
    - **Benefit**: Allows the JetBot to make intelligent decisions based on visual data without requiring external computation resources.

3. **Sensor Data Processing**:
    - **Description**: Processes data from the TCS3200 color sensor to detect specific colors.
    - **Implementation**:
        - Reads sensor data and determines the presence and intensity of the color red.
        - Uses this information to influence the JetBot’s behavior, such as navigating or stopping.
    - **Benefit**: Enhances the JetBot’s ability to interact with its environment by detecting and responding to color cues.

4. **Wireless Connectivity**:
    - **Description**: Provides wireless connectivity for remote control and monitoring.
    - **Implementation**:
        - Integrates a WiFi module to enable communication over a network.
        - Uses Flask-based web services to handle remote commands and data exchange.
    - **Benefit**: Allows users to control and monitor the JetBot from a distance, making it more versatile and easier to manage.


## Installation
Steps required to install and set up the project. Include any prerequisites, dependencies, and commands needed to get the project running.
## Installation

### Prerequisites

Before you begin, ensure you have met the following requirements:

- **Operating System**: The project is designed to run on Ubuntu-based systems, specifically on the NVIDIA Jetson Nano.
- **Python**: Ensure Python 3.6 or higher is installed.
- **NVIDIA Jetson Nano**: Make sure you have a Jetson Nano with JetPack SDK installed.
- **JetBot Kit**: Assemble the JetBot kit according to the instructions provided with the kit.

### Dependencies

Install the necessary dependencies by running the following commands:

1. **Update and Upgrade Your System**:
    ```bash
    sudo apt-get update
    sudo apt-get upgrade
    ```

2. **Install Python Packages**:
    ```bash
    sudo apt-get install python3-pip
    pip3 install jetbot
    pip3 install torch torchvision
    pip3 install opencv-python
    pip3 install pillow
    pip3 install requests
    pip3 install flask
    pip3 install numpy
    ```

3. **Install GStreamer**:
    ```bash
    sudo apt-get install libgstreamer1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools
    ```

4. **Install Additional Libraries for the TCS3200 Color Sensor**:
    ```bash
    pip3 install adafruit-circuitpython-tcs34725
    ```

### Setting Up the Project

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/ELE495-2324Summer/capstoneproject-corr.git
    ```

2. **Load the Pre-trained Model**:
    - Download the pre-trained PyTorch model file (`70x70_14.07.pt`) and place it in the project directory.

3. **Set Up Flask Server**:
    - Navigate to the server directory and run the Flask server.
    ```bash
    cd jetbot/notebooks/basic_motion
    python3 app.py
    ```

4. **Run the Main Script**:
    - Ensure all hardware components are connected and powered on.
    - Run the main script to start the JetBot application.
    ```bash
    python3 kirmiziya_girme.py
    ```

### Mobile Application

If you have a mobile application, follow these steps to set it up:

1. **Install the Mobile App**:
    - Download the mobile application from the relevant app store or repository.
    - Install the application on your mobile device.

2. **Connect to JetBot**:
    - Ensure your mobile device is connected to the same WiFi network as the JetBot.
    - Open the app and follow the instructions to connect to the JetBot.
    - Write your IP address and connect to JetBot
3. **Useage of App**:
    - Send your plate number which you want to park
    - Then, wait for the process
    - Repeat the steps

### Troubleshooting

If you encounter any issues during installation or setup, refer to the troubleshooting section in the documentation or check the project's issue tracker on GitHub.

## Usage
```bash
ssh -x jetbot@192.168.55.1 #Jetson Nano must be connected to PC with USB
#Then, you must sign in with password which is 'jetbot'.
sudo nmcli device wifi connect 'SSID' password 'password' #And, you are ready for wireless connection
cd jetbot/notebooks/basic_motion
python3 app.py #It starts flask server to communicate mobil app between JetBot
#Then, You should open a new command prompt, You should repeat the steps above except wifi connetion
python3 kirmiziya_girme.py #it start the main code, Finally, you are ready to go.
```
## Screenshots
![Final Appearance of the JetBot](jetbot%20fotolar/WhatsApp%20Image%202024-07-25%20at%2015.42.43.jpeg)


[Contributor 1](https://github.com/user1)
[Resource or Tool](https://www.nvidia.com)
