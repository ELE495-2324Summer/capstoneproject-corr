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
  - **json**: Used for handling JSON data.
- PyTorch     : An open-source machine learning library used for developing the digit recognition model.
- GStreamer   : A pipeline-based multimedia framework used to handle video streams.
### Applications 
### Services 

## Installation
Describe the steps required to install and set up the project. Include any prerequisites, dependencies, and commands needed to get the project running.

```bash
# Example commands
git clone https://github.com/username/project-name.git
cd project-name
```

## Usage
Provide instructions and examples on how to use the project. Include code snippets or screenshots where applicable.

## Screenshots
Include screenshots of the project in action to give a visual representation of its functionality. You can also add videos of running project to YouTube and give a reference to it here. 

## Acknowledgements
Give credit to those who have contributed to the project or provided inspiration. Include links to any resources or tools used in the project.

[Contributor 1](https://github.com/user1)
[Resource or Tool](https://www.nvidia.com)
