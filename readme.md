# Face Mask Usage Detection Using Inception Network

The main goal of this code is to detect human faces and classify those faces if it is wearing a face mask or covering properly.

## Installation Instructions
* Make sure you have installed Python 3.6 or newer
* Go to the directory/folder where you downloaded the code then press Shift+Right Click then click to open PowerShell/Terminal/Command Prompt.
* Install pip: `python get-pip.py`
* Install tensorflow: `pip install --upgrade tensorflow`
* Then install the rest: `pip install requirements.txt`

## Testing the Trained Model
* To test the model using a webcam: `python test.py`
* To test the model using a YouTube video: `python test.py YOUTUBE_URL` example: `python test.py https://www.youtube.com/watch?v=Ft62ShND99Q`