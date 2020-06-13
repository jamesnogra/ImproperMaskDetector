import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import sys
import pafy

import settings
from model import *
from serve import get_model_api

min_pixel_face = 16 #minimum size of face in pixels to be detected
model_api = get_model_api()
frames_to_process = 4 #how many times in one second do we process faces

try:
    mirror = False
    source = sys.argv[1]
except IndexError:
    source = "webcam"
    mirror = True


def show_webcam(mirror, source):
    print("\n\n\nPress 'q' to quit...")
    at_frame = 0
    
    if (source=="webcam"):
        print("Using webcam as source of video...")
        cap = cv2.VideoCapture(0)
    elif "youtube" in source:
        print("Using YouTube video as source of video...")
        video_pafy = pafy.new(source)
        video_from_url = video_pafy.getbest().url
        cap = cv2.VideoCapture(video_from_url)
    else:
        print("Using video file " + source + " as source of video...")
        cap = cv2.VideoCapture(source)

    #determine the resolution and FPS of the source
    video_width  = cap.get(3)  # width of the video
    video_height = cap.get(4) # height of the video
    video_fps = cap.get(5) #FPS (frame rate) of the source video
    skip_frames = int(video_fps/frames_to_process)
    print("Video source is "+str(video_width)+"x"+str(video_height)+ " at "+str(video_fps)+" FPS.")

    faces = [] #empty detected faces on initialize
    while True:
        ret_val, frame = cap.read()
        if (not ret_val):
            exit()
        #check the dimensions if needed to resize for performance
        if (video_height>=720): #if video is large, then we resize it before processing it
            frame = cv2.resize(frame, (int(video_width*0.65), int(video_height*0.65)))
        #only webcam has a mirror=True
        if mirror: 
            frame = cv2.flip(frame, 1)
        #only process frames at a specific frame
        if (at_frame%skip_frames==0):
            #convert the image to gray for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #detect the faces
            faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(min_pixel_face, min_pixel_face)
            )
            at_frame = 0 #reset the frames
        #draw squares on faces
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
            #only draw square if the classification is not `not_face`
            if (face_class!="not_face"):
                cv2.putText(frame, face_class, (x, y-5), 0, 0.5, color, 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            #cv2.rectangle(frame, (0, 0), (int(frame.shape[1]), int(frame.shape[0])), color, 10) #draw full rectangle in border
        cv2.imshow('frame', frame) #uncomment this to see live tracking
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break  # q to quit
        at_frame += 1
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror, source)


if __name__ == '__main__':
    main()