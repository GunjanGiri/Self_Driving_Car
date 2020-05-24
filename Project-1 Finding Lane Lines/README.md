# **Project 1: Finding Lane Lines on the Road** 

## Introduction

For a self driving vehicle to stay in a lane, the first step is to identify lane lines before issuing commands to the control system. Since the lane lines can be of different colors (white, yellow) or forms (solid, dashed) this seemingly trivial task becomes increasingly difficult. Moreover, the situation is further exacerbated with variations in lighting conditions. Thankfully, there are a number of mathematical tools and approaches available nowadays to effectively extract lane lines from an image or dashcam video. In this project, a program is written in python to identify lane lines on the road, first in an image, and later in a video stream. After a thorough discussion of the methodology, potential shortcomings and future improvements are suggested.

## Methodology
Before attempting to detect lane lines in a video, a software pipeline is developed for lane detection in a series of images. Only after ensuring that it works satisfactorily for test images, the pipeline is employed for lane detection in a video. 
The pipeline consisted of 5 major steps excluding reading and writing the image. 

1. First we will create Region of Interest for the Videos and cover the masks for the Lane and the form the the line by calling Draw the Lines function .

2. And then we will convert the image into gray_image and then smoothen the image using Gaussian Blur and then use Canny Edge Detection to form Canny Edge and then form the Cropped Image.

3. Another improvement in the pipeline can be to include a running average of slopes for identified lane lines so that there is a smooth transition from one frame to the next. This avoids rapid changes in commands to the steering control system.

4. Further, machine learning approaches can be explored to make the lane finding pipeline more robust in future.
