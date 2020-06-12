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
skip_frames = 5

try:
    video_file = sys.argv[1]
except IndexError:
    print("Provide a valid video file from the videos folder. Example: `python datagather-video.py test1.mp4`.")
    exit()


def show_video(video_file):
    print("\n\n\nPress 'q' to quit...")
    at_frame = 0
    
    cap = cv2.VideoCapture("videos/"+video_file)
    video_width  = cap.get(3)  # width of the video
    video_height = cap.get(4) # height of the video
    video_fps = cap.get(5) #FPS of the source video
    skip_frames = int(video_fps/3)
    faces = []

    while True:
        ret_val, frame = cap.read()
        if (not ret_val):
            exit()
        if (video_height>=720):
            frame = cv2.resize(frame, (int(video_width*0.5), int(video_height*0.5)))
        if (at_frame%skip_frames==0):
            #convert the image to gray for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #detect the faces
            faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_pixel_face, min_pixel_face)
            )
            at_frame = 0 #reset the frames
        for (x, y, w, h) in faces:
            #crop each face
            crop_img = gray[y: y + h, x: x + w]
            crop_img = cv2.resize(crop_img, (settings.IMG_SIZE, settings.IMG_SIZE))
            face_class, probability = model_api(crop_img)
            probability = " " + str(round(probability*100, 1)) + "%"
            color = (0, 255, 0)
            if (face_class=="not_covered"):
                color = (0, 0, 255)
            elif (face_class=="partially_covered"):
                color = (51, 153, 255)
            if (face_class!="not_face"):
                cv2.putText(frame, face_class, (x, y-5), 0, 0.5, color, 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        cv2.imshow('frame', frame) #uncomment this to see live tracking
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break  # q to quit
        at_frame += 1
    cv2.destroyAllWindows()


def main():
    show_video(video_file)


if __name__ == '__main__':
    main()