import cv2, pafy
import random, string, sys

frames_to_skip = 10
saved_image_size = 64 #size of the saved faces in pixels
min_pixel_face = 16 #minimum size of face in pixels to be detected
at_frame = 0
frames_to_process = 4 #how many times in one second do we process faces

try:
	url = sys.argv[1]
except IndexError:
	print("Provide a valid YouTube URL or a valid video file. Example: `python datagather.py https://www.youtube.com/watch?v=P27HRClMf2U` OR `python datagather.py videos/test1.mp4`")
	exit()

#this is to generate filenames of faces
def randomString(stringLength=8):
	letters = string.ascii_lowercase
	return ''.join(random.choice(letters) for i in range(stringLength))

if (url=="webcam"):
	frames_to_skip = 1
	cap = cv2.VideoCapture(0)
else:
	if "youtube" in url:
		print("Using YouTube video as source of video...")
		video_pafy = pafy.new(url)
		video_from_url = video_pafy.getbest().url
		cap = cv2.VideoCapture(video_from_url)
	else:
		print("Using video file " + url + " as source of video...")
		cap = cv2.VideoCapture(url)

#determine the resolution and FPS of the source
video_width  = cap.get(3)  # width of the video
video_height = cap.get(4) # height of the video
video_fps = cap.get(5) #FPS (frame rate) of the source video
skip_frames = int(video_fps/frames_to_process)
print("Video source is "+str(video_width)+"x"+str(video_height)+ " at "+str(video_fps)+" FPS.")

#get the frames of the video
while (True):
	ret, frame = cap.read()
	if (not ret):
		exit()
	#check the dimensions if needed to resize for performance
	if (video_height>=720): #if video is large, then we resize it before processing it
		frame = cv2.resize(frame, (int(video_width*0.65), int(video_height*0.65)))
	#check if we need to check this frame for faces
	if (at_frame%skip_frames==0):
		#convert the image to gray for analysis
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#detect the faces
		faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
		faces = faceCascade.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=2,
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
		at_frame = 0
	#draw rectangles in faces
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
	cv2.imshow('frame', frame) #uncomment this to see live tracking
	#to exit, press 'q'
	if cv2.waitKey(5) & 0xFF == ord('q'): #press `q` to quit
		break
	if (not ret):
		break
	at_frame += 1

cap.release()
cv2.destroyAllWindows()
print("Done scanning faces in "+url+".")