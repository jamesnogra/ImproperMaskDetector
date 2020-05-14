import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import sys

import settings
from model import *
from serve import get_model_api

min_pixel_face = 16 #minimum size of face in pixels to be detected
model_api = get_model_api()


def show_webcam(mirror=False):
    print("\n\n\nPress 'q' to quit...")
    cap = cv2.VideoCapture(0)
    while True:
        ret_val, frame = cap.read()
        if mirror: 
            frame = cv2.flip(frame, 1)
        #convert the image to gray for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #detect the faces
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(min_pixel_face, min_pixel_face)
        )
        for (x, y, w, h) in faces:
            #crop each face
            crop_img = gray[y: y + h, x: x + w]
            crop_img = cv2.resize(crop_img, (settings.IMG_SIZE, settings.IMG_SIZE))
            face_class = model_api(crop_img)
            color = (0, 255, 0)
            if (face_class=="not_covered"):
                color = (0, 0, 255)
            elif (face_class=="partially_covered"):
                color = (51, 153, 255)
            cv2.putText(frame, face_class, (x, y-5), 0, 0.5, color, 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        cv2.imshow('frame', frame) #uncomment this to see live tracking
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break  # q to quit
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()