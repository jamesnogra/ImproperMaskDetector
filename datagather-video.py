import cv2
import random, string, sys

frames_to_skip = 50
saved_image_size = 64 #size of the saved faces in pixels
min_pixel_face = 16 #minimum size of face in pixels to be detected

try:
	video_file = sys.argv[1]
except IndexError:
	print("Provide a valid video file from the videos folder. Example: `python datagather-video.py test1.mp4`.")
	exit()

#this is to generate filenames of faces
def randomString(stringLength=8):
	letters = string.ascii_lowercase
	return ''.join(random.choice(letters) for i in range(stringLength))

frames_to_skip = 5
at_frame = 0
cap = cv2.VideoCapture("videos/"+video_file)
video_width  = cap.get(3)  # float
video_height = cap.get(4) # float

#get the frames of the video
while (True):
	ret, frame = cap.read()
	if (not ret):
		exit()
	if (video_height>=720):
		frame = cv2.resize(frame, (int(video_width*0.65), int(video_height*0.65)))
	#check if we need to check this frame for faces
	if at_frame == 0:
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
		#save rectangles in faces
		for (x, y, w, h) in faces:
			#save each face to the untagged folder
			crop_img = frame[y: y + h, x: x + w]
			crop_img = cv2.resize(crop_img, (saved_image_size, saved_image_size))
			crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
			temp_filename = "untagged/" + randomString() + ".jpg"
			cv2.imwrite(temp_filename, crop_img)
			print("Face saved at " + temp_filename)
	#draw rectangles in faces
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
	cv2.imshow('frame',frame) #uncomment this to see live tracking
	#to exit, press 'q'
	if cv2.waitKey(20) & 0xFF == ord('q'): #press `q` to quit
		break
	if (not ret):
		break
	at_frame += 1
	if (at_frame>frames_to_skip):
		at_frame = 0

cap.release()
cv2.destroyAllWindows()
print("Done scanning faces in "+video_file+".")