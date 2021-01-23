# EL2320 course project
# Author: Michaela Söderström

"""
Help function for finding frames.
------------------------------------
"""

# Load libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

def findROI(frame):
    '''
    Target identification of a human frontal face. Detect a target using pretrained Haar Cascade detection.

    data source: opencv.org
    ---------------------------------
    INPUT:
        frame: 
            the desired image to detect a target on if it extists
    OUTPUT:
        coord: list
            contains the x, y top left coordinates and the width and height of the target region
        frame_cropped: array
            coordinates of the roi area when target is detected
        frame_color: 
            rectangle drawn on the target area of the original input frame
    ---------------------------------
    '''
    coord = () 
    frame_cropped = () # create empty tuple
    frame_color = np.copy(frame)

    face_cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # make gray-scale
    frame_gray = cv2.equalizeHist(frame_gray)

    detected = face_cascade.detectMultiScale(frame_gray,
                                            scaleFactor=1.5)
    coord = list(zip(*detected))

    for (x, y, w, h) in detected:
        frame_color = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3) # draw a rectangle around the target
        frame_cropped = frame[y: y + h, x: x + w] # crop the frame to fit roi

    return coord, frame_cropped, frame_color