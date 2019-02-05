# DigitRecognition
DigitRecognition using keras/SVC and pygame.

# Overview
Recently Deep Convolutional Neural Networks (CNNs) becomes one of the most appealing approaches and has been a crucial factor in the variety of recent success and challenging machine learning applications such as object detection, and face recognition. Therefore, CNNs is considered our main model for our challenging tasks of image classification. Specifically, it is used for is one of high research and business transactions. Handwriting digit recognition application is used in different tasks of our real-life time purposes. Precisely, it is used in vehicle number plate detection, banks for reading checks, post offices for sorting letter, and many other related tasks.

# Description
This is a DigitRecognition application which can predict output corresponding to handwritten images. I used SVC(support vector classifier) and sequential model of Keras for creating this predictive model. I trained SVC for 8X8 MNIST dataset, but the accuracy of this model is not good when I run this model on my handwritten images(600X600). It is due to resizing images from 600X600 to 8X8.It is important to get good results so I created a sequential model in keras and traied it on 28X28 MNIST dataset. Now it gives very good result on handwritten digits. 

The interface is created by using Pygame. The image preprocessing is the most important in this project which I have done by using Scipy and OpenCV.

# Dataset
MNIST is a widely used dataset for the hand-written digit classification task. It consists of 70,000 labelled 28x28 pixel grayscale images of hand-written digits. The dataset is split into 60,000 training images and 10,000 test images. There are 10 classes (one for each of the 10 digits). The task at hand is to train a model using the 60,000 training images and subsequently test its classification accuracy on the 10,000 test images.

## Sample Images:
These are some sample images of the handwritten character from mnist dataset. <br><br>
	![sample images](assets/sample_images.png "images in mnist dataset")<br><br>

# Dependencies
This is the list of dependencies for running this application.

* **Skleran**
 * **Keras**
 * **tensorflow/theano**
 * **Opencv**
 * **Pygame**
 * **Pandas**
 * **Numpy**
 * **Scipy**
 * **Matplotlib**

# Multi digit recognition
I am developing an efficient model for detection multiple digits on a single frame like number plate, phone number, cheque number etc. 
Here are some results:<br><br>
	![Pygame window](assets/Capture1.PNG "multi digits" )<br><br>

# Execution	
To run the code, type python app.py

# Digit Recognition using Keras/TensorFlow and OpenCV in Python

# Description
I wanted to use what 've learned on DeepLearning to make a real-world project. It was fun and i've learned many new things.

# Python Implementation

* **Dataset- MNIST dataset**
* **Images of size 28 X 28**
* **Classify digits from 0 to 9**
* **CNN.**

# Train Acuracy 99,45%
# Test Acuracy 99,32%
<br><br>
	![Capture2](assets/Capture2.PNG "images in mnist dataset")<br><br>

# Execution for showing images through webcam
To run the code, type python cam.py
