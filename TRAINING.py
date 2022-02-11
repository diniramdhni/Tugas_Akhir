#import library
import numpy
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D,\
                                     Dense, Input, Dropout, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

train_directory = 'data/train/'
for expression in os.listdir(train_directory):
  print(f"{expression} = {len(os.listdir(train_directory+expression))} images")

#data augmentation
img_size, batch_size = 48, 48
data_gen = ImageDataGenerator(horizontal_flip = True)
train_generator = data_gen.flow_from_directory('data/train/',
                                         target_size = (img_size, img_size),
                                         color_mode = 'grayscale',
                                         class_mode = 'categorical',
                                         batch_size = batch_size,
                                         shuffle = True)

validation_generator = data_gen.flow_from_directory('data/test/',
                                         target_size = (img_size, img_size),
                                         color_mode = 'grayscale',
                                         class_mode = 'categorical',
                                         batch_size = batch_size,
                                         shuffle = True)

#model cnn
model = Sequential()

#1-Conv Layer
model.add(Conv2D(64, (3,3), padding = 'same', input_shape = (48, 48, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

#2-Conv Layer
model.add(Conv2D(128, (5,5), padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

#3-Conv Layer
model.add(Conv2D(512, (3,3), padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

#4-Conv Layer
model.add(Conv2D(512, (3,3), padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(7, activation = 'softmax'))


#compile model
model.compile(optimizer = Adam(lr=0.00005), loss = 'categorical_crossentropy', metrics = ['accuracy'])

#training model dengan fit_generator
epochs = 375
train_steps = train_generator.n//train_generator.batch_size
validation_steps = validation_generator.n//validation_generator.batch_size

#menyimpan bobot saat training
checkpoint = ModelCheckpoint('model_weights.h5',monitor = 'val_accuracy',save_weights_only = True,mode = 'max',verbose = 1)

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',factor = 0.1,patience = 2,min_lr = 0.00001,mode = 'auto')

callbacks = [checkpoint, reduce_lr]

history = model.fit_generator(train_generator,
                    steps_per_epoch = train_steps,
                    epochs = epochs,
                    validation_data = validation_generator,
                    validation_steps = validation_steps,
                    callbacks = callbacks)

#simpan model
model_json = model.to_json()
with open('model.json', 'w') as json_file:
  json_file.write(model_json)

