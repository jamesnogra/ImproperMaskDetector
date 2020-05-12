import cv2, pafy
import random, string, sys

frames_to_skip = 50
saved_image_size = 64 #size of the saved faces in pixels
min_pixel_face = 16 #minimum size of face in pixels to be detected

try:
	url = sys.argv[1]
except IndexError:
	print("Provide a valid YouTube URL. Example: `python datagather.py https://www.youtube.com/watch?v=P27HRClMf2U`")
	exit()

#this is to generate filenames of faces
def randomString(stringLength=8):
	letters = string.ascii_lowercase
	return ''.join(random.choice(letters) for i in range(stringLength))

video_pafy = pafy.new(url)
video_from_url = video_pafy.getbest().url
cap = cv2.VideoCapture(video_from_url)

#get the frames of the video
while (True):
	#get the frame number
	frame_id = int(round(cap.get(1)))
	#print("Getting frame "+str(frame_id)+"...")
	#read frame by frame
	ret, frame = cap.read()
	#check if we need to check this frame for faces
	if frame_id % frames_to_skip == 0:
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
		#draw rectangles in faces
		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			#save each face to the untagged folder
			crop_img = frame[y: y + h, x: x + w]
			crop_img = cv2.resize(crop_img, (saved_image_size, saved_image_size))
			crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
			temp_filename = "untagged/" + randomString() + ".jpg"
			cv2.imwrite(temp_filename, crop_img)
			print("Face saved at " + temp_filename)
		#cv2.imshow('frame',frame) #uncomment this to see live tracking
	#to exit, press 'q'
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
	if (not ret):
		break

cap.release()
cv2.destroyAllWindows()
print("Done scanning faces in "+url+".")