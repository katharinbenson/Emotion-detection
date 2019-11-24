# Emotion-detection

## Introduction

This project aims to classify the emotion on a person's face into one of the 7 categories i.e. angry, disgusted, fearful, happy, neutral, sad and surprised using deep convolutional neural networks.

## Dependencies

1. Python 3.6
2. OpenCV
3. Keras
4. Tensorflow

## Data Source

I used the [Face expression recognition dataset](https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset) which has 2 folders, train and validation. Each of these folders have 7 subfolders of emotions - angry, disgusted, fearful, happy, neutral, sad and surprised.

## Video Analysis

The real time video processing was bulit in the following way:

1. Launch the webcam
2. Identify the face by haar cascade
3. Zoom the model and find the roi(region of intrest), which is a face in this case.
3. Dimension the face to 48 * 48 pixels.
4. Make a prediction on the face using our pre-trained model.
5. The network outputs a list of softmax scores for the seven classes.
6. The emotion with maximum score is displayed on the screen.

## Examples

![alt text](https://github.com/katharinbenson/Emotion-detection/blob/master/happy.png)
![alt text](https://github.com/katharinbenson/Emotion-detection/blob/master/angry.png)
![alt text](https://github.com/katharinbenson/Emotion-detection/blob/master/fear.png)
![alt text](https://github.com/katharinbenson/Emotion-detection/blob/master/disgust.png)

