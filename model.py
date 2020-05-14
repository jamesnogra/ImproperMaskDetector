import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression

import settings

input = input_data(shape=[None, settings.IMG_SIZE, settings.IMG_SIZE, settings.IMAGE_CHANNELS], name='input')

#tower 1 is a 1x1 convolution followed by a 3x3 convolution
tower_1_1 = conv_2d(input, settings.FIRST_NUM_CHANNEL, 1, padding='same', activation='relu')
tower_1_1 = conv_2d(tower_1_1, settings.FIRST_NUM_CHANNEL, 3, padding='same', activation='relu')

#tower 2 is a 1x1 convolution followed by a 5x5 convolution
tower_1_2 = conv_2d(input, settings.FIRST_NUM_CHANNEL, 1, padding='same', activation='relu')
tower_1_2 = conv_2d(tower_1_2, settings.FIRST_NUM_CHANNEL, 5, padding='same', activation='relu')

#tower 3 is just a 1x1 convolution
tower_1_3 = conv_2d(input, settings.FIRST_NUM_CHANNEL*2, 1, padding='same', activation='relu')

#the first inception layer is still a IMG_SIZExIMG_SIZE
inception_1 = merge([tower_1_1, tower_1_2, tower_1_3], mode='concat', axis=3, name='Merge')
print("Inception 1 Shape: ", inception_1.get_shape())

#resize dimension to half
inception_1 = max_pool_2d(inception_1, 2)

#tower 1 is a 1x1 convolution followed by a 3x3 convolution
tower_2_1 = conv_2d(inception_1, settings.FIRST_NUM_CHANNEL*2, 1, padding='same', activation='relu')
tower_2_1 = conv_2d(tower_2_1, settings.FIRST_NUM_CHANNEL*2, 3, padding='same', activation='relu')

#tower 2 is a 1x1 convolution followed by a 5x5 convolution
tower_2_2 = conv_2d(inception_1, settings.FIRST_NUM_CHANNEL*2, 1, padding='same', activation='relu')
tower_2_2 = conv_2d(tower_2_2, settings.FIRST_NUM_CHANNEL*2, 5, padding='same', activation='relu')

#tower 3 is just a 1x1 convolution
tower_2_3 = conv_2d(inception_1, settings.FIRST_NUM_CHANNEL*4, 1, padding='same', activation='relu')

#the second inception layer is now a (IMG_SIZE/2)x(IMG_SIZE/2) image
inception_2 = merge([tower_2_1, tower_2_2, tower_2_3], mode='concat', axis=3, name='Merge')
print("Inception 2 Shape: ", inception_2.get_shape())

#resize dimension to half
inception_2 = max_pool_2d(inception_2, 2)

#tower 1 is a 1x1 convolution followed by a 3x3 convolution
tower_3_1 = conv_2d(inception_2, settings.FIRST_NUM_CHANNEL*4, 1, padding='same', activation='relu')
tower_3_1 = conv_2d(tower_3_1, settings.FIRST_NUM_CHANNEL*4, 3, padding='same', activation='relu')

#tower 2 is a 1x1 convolution followed by a 5x5 convolution
tower_3_2 = conv_2d(inception_2, settings.FIRST_NUM_CHANNEL*4, 1, padding='same', activation='relu')
tower_3_2 = conv_2d(tower_3_2, settings.FIRST_NUM_CHANNEL*4, 5, padding='same', activation='relu')

#tower 3 is just a 1x1 convolution
tower_3_3 = conv_2d(inception_2, settings.FIRST_NUM_CHANNEL*8, 1, padding='same', activation='relu')

#the third inception layer is now a (IMG_SIZE/4)x(IMG_SIZE/4) image
inception_3 = merge([tower_3_1, tower_3_2, tower_3_3], mode='concat', axis=3, name='Merge')
print("Inception 3 Shape: ", inception_3.get_shape())

#resize dimension to half
inception_3 = max_pool_2d(inception_3, 2)

output = fully_connected(inception_3, settings.FIRST_NUM_CHANNEL*16, activation='relu')
output = dropout(output, 0.8)
print("FC Shape: ", output.get_shape())