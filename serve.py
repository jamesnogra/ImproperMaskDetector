import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import sys
import json
from operator import itemgetter
import base64

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import settings
from model import *

#getting all folders
def define_classes():
	print('Defining classes...')
	all_classes = []
	for folder in tqdm(os.listdir(settings.TRAIN_DIR)):
		all_classes.append(folder)
	return all_classes, len(all_classes)

#define labels using the folders
def define_labels(all_classes):
	print('Defining labels...')
	all_labels = []
	for x in tqdm(range(len(all_classes))):
		all_labels.append(np.array([0. for i in range(len(all_classes))]))
		all_labels[x][x] = 1.
	return all_labels

all_classes, settings.NUM_OUTPUT = define_classes()
all_labels = define_labels(all_classes)

def get_model_api():
	#other model definition is at model.py
	loss = fully_connected(output, settings.NUM_OUTPUT, activation='softmax')
	network = tflearn.regression(loss, optimizer='RMSprop', loss='categorical_crossentropy', learning_rate=settings.LR, name='targets')
	model = tflearn.DNN(network, checkpoint_path=settings.MODEL_NAME, max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir="./tflearn_logs/")
	print('LOADING MODEL:', '{}.meta'.format(settings.MODEL_NAME))
	if os.path.exists('{}.meta'.format(settings.MODEL_NAME)):
		model.load(settings.MODEL_NAME)
		print('Model', settings.MODEL_NAME,'loaded...')

	def classify_face(temp_img):
		data = temp_img.reshape(settings.IMG_SIZE, settings.IMG_SIZE, settings.IMAGE_CHANNELS)
		data_res_float = model.predict([data])[0]
		max_probability = max(data_res_float)
		data_res = np.round(data_res_float, 0)
		str_label = '?'
		for x in range(len(all_labels)):
			#print(all_classes[x], str(round((data_res_float[x]*100),4)),'%')
			#print("Comparing:", data_res, " and ", all_labels[x])
			if ((data_res==all_labels[x]).all()):
				str_label = all_classes[x]
		return str_label, max_probability

	return classify_face