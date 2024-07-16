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

Overall, PyTorch provides a robust and flexible framework for developing and deploying machine learning models, and its use in this project showcases its capabilities in a real-world application involving image recognition and robotic control.

- GStreamer   : GStreamer is a pipeline-based multimedia framework used to handle video streams. In this project, GStreamer is utilized to capture video input from a camera and process it in real-time.
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
