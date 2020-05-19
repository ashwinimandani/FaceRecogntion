import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.resnet50 import ResNet50 
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from PIL import Image
import base64
import random
from keras.models import load_model
import cv2
from io import BytesIO

resnet = ResNet50(input_shape=Image + [3], weights='imagenet', include_top=False)

for layer in resnet.layers:
  layer.trainable = False  

x = Flatten()(resnet.output)
x = Dense(1000, activation='relu')(x)
no_of_classifier=2
prediction = Dense(no_of_classifier, activation='softmax')(x)

model = Model(inputs=resnet.input, outputs=prediction)
model.summary()
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
training_data_gen=ImageDataGenerator(rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            rescale=1./255,
                            fill_mode="nearest")

test_data_gen=ImageDataGenerator(rescale=1.0/255)

training_set = training_data_gen.flow_from_directory('your_training_dataset_path',
                                                 target_size = (299, 299),
                                                 batch_size = 16,
                                                 class_mode = 'categorical')

test_set = test_data_gen.flow_from_directory('your_testing_dataset_path',
                                            target_size = (299, 299),
                                            batch_size = 16,
                                            class_mode = 'categorical')


r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

from keras.models import load_model

model.save('facerecognition.h5')
