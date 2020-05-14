import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import matplotlib.pyplot as plt

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

def create_train_data(all_classes, all_labels):
	training_data = []
	label_index = 0
	for specific_class in all_classes:
		current_dir = settings.TRAIN_DIR + '/' + specific_class
		print('Loading all', all_classes[label_index], 'images...')
		for img in tqdm(os.listdir(current_dir)):
			path = os.path.join(current_dir,img)
			if (settings.IMAGE_CHANNELS==1):
				img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
			elif (settings.IMAGE_CHANNELS==3):
				img = cv2.imread(path)
			img = cv2.resize(img, (settings.IMG_SIZE, settings.IMG_SIZE))
			training_data.append([np.array(img)/255, np.array(all_labels[label_index])]) #we normalize the images hence the /255
		label_index = label_index + 1
	shuffle(training_data)
	return training_data

all_classes, settings.NUM_OUTPUT = define_classes()
all_labels = define_labels(all_classes)
training_data = create_train_data(all_classes, all_labels)

#define the training data and test/validation data
train = training_data[:int(len(training_data)*0.8)] #80% of the training data will be used for training
test = training_data[-int(len(training_data)*0.2):] #20% of the training data will be used for validation
X = np.array([i[0] for i in train]).reshape(-1, settings.IMG_SIZE, settings.IMG_SIZE, settings.IMAGE_CHANNELS)
Y = [i[1] for i in train]
test_x = np.array([i[0] for i in test]).reshape(-1, settings.IMG_SIZE, settings.IMG_SIZE, settings.IMAGE_CHANNELS)
test_y = [i[1] for i in test]


#other model definition is at model.py
loss = fully_connected(output, settings.NUM_OUTPUT, activation='softmax')
network = tflearn.regression(loss, optimizer='RMSprop', loss='categorical_crossentropy', learning_rate=settings.LR, name='targets')
model = tflearn.DNN(network, checkpoint_path=settings.MODEL_NAME, max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir="./tflearn_logs/")


model.fit({'input': X}, {'targets': Y}, n_epoch=settings.NUM_EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), batch_size=64, snapshot_step=5000, show_metric=True, run_id=settings.MODEL_NAME)
#model.fit({'input': X}, {'targets': Y}, n_epoch=NUM_EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), shuffle=True, show_metric=True, batch_size=32, snapshot_step=500, snapshot_epoch=False, run_id=MODEL_NAME)


model.save(settings.MODEL_NAME)
print('MODEL SAVED:', settings.MODEL_NAME)


#validate and plot
fig=plt.figure()
for num,data in enumerate(training_data[:12]):
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(settings.IMG_SIZE, settings.IMG_SIZE, settings.IMAGE_CHANNELS)
    
    data_res = np.round(model.predict([data])[0], 0)
    str_label = '?'
    for x in range(len(all_labels)):
    	#print("Comparing:", data_res, " and ", all_labels[x])
    	if ((data_res==all_labels[x]).all()):
    		str_label = all_classes[x]
    		#print("Label is", str_label)
        
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()