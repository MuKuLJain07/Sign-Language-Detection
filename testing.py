import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import os

##################################
offset = 20
imgSize = 300
Directory = r'Dataset/B'
counter = 1
lqbels = ['A', 'B']

# Ensure directory exists
if not os.path.exists(Directory):
    os.makedirs(Directory)
##################################

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

while True: 
    ret, frame = cap.read() 
    hands, frame = detector.findHands(frame)   

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Ensuring the region is within the frame boundaries
        imgCrop = frame[max(0, y-offset): min(y+h+offset, frame.shape[0]), max(0, x-offset): min(x+w+offset, frame.shape[1])]

        aspectRatio = h / w

        if aspectRatio > 1:  # vertical rectangle
            k = imgSize / h
            wGap = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wGap, imgSize))
            wGap = math.ceil((imgSize - wGap) / 2)
            imgWhite[:, wGap:wGap + imgResize.shape[1]] = imgResize
            prediction, index = classifier.getPrediction(frame)
            print(prediction, index)

        else:  # horizontal rectangle
            k = imgSize / w
            hGap = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hGap))
            hGap = math.ceil((imgSize - hGap) / 2)
            imgWhite[hGap:hGap + imgResize.shape[0], :] = imgResize

        # Display the resized image
        cv2.imshow('imageWhite', imgWhite)

    # Display the original frame
    cv2.imshow('frame', frame)
    
    # Collect data and save image when 's' is pressed
    key = cv2.waitKey(1)

    # Exit on 'q' key press
    if key == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
