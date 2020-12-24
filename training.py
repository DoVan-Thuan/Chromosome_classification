import keras
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout, Activation, BatchNormalization, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import cv2
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from sklearn.utils import class_weight

train_direction = './data/train'
test_direction = './data/test'
valid_direction = './data/valid'

train_gen = ImageDataGenerator(rotation_range=45, zoom_range=0.2,
                               horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
valid_gen = ImageDataGenerator()

batch_size = 128
training_data = train_gen.flow_from_directory(directory=train_direction,
                                              target_size=(128, 128),
                                              batch_size=batch_size,
                                              class_mode='binary'
                                              )
valid_data = valid_gen.flow_from_directory(directory=valid_direction,
                                           target_size=(128, 128),
                                           batch_size=batch_size,
                                           class_mode='binary'
                                           )

from keras.applications import VGG16
vgg = VGG16(weights='imagenet', include_top=False, input_shape=training_data.image_shape)

for layer in vgg.layers[15:-1]:
    layer.trainable = False

vgg.layers[-1].trainable=False
vgg.layers[0].trainable=True
for layer in vgg.layers:
    print(layer, layer.trainable)

vgg.summary()

model = Sequential()

# Finetune from VGG:
model.add(vgg)

# Add new layers
model.add(Dropout(rate=0.55))
model.add(Conv2D(filters=len(set(training_data.classes)), kernel_size=(3, 3), padding='same'))
model.add(GlobalAveragePooling2D())
model.add(BatchNormalization())
model.add(Activation('softmax'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.RMSprop(lr=1.5e-4), metrics=['acc'])
with tensorflow.device('/GPU:0'):
  history = model.fit(training_data,
                        steps_per_epoch = int(24268/batch_size),
                        epochs = 50,
                        validation_data = valid_data,
                        validation_steps = int(8348/batch_size))

test_data = valid_gen.flow_from_directory(directory=test_direction,
                                            target_size=(128,128),
                                            batch_size=batch_size,
                                            class_mode='binary',
                                          shuffle = False)
from sklearn.metrics import classification_report, confusion_matrix
Y_pred = model.predict(test_data)
y_pred = np.argmax(Y_pred, axis=1)
print(test_data.classes[100:500])
print(y_pred[100:500])

print(Y_pred.shape)
print('Confusion Matrix')
print(confusion_matrix(test_data.classes, y_pred))
print('Classification Report')
print(classification_report(test_data.classes, y_pred))

#unfreeze
for layer in vgg.layers:
    layer.trainable = True

# vgg.layers[-1].trainable=False

for layer in vgg.layers:
    print(layer, layer.trainable)

model.compile(optimizer = tensorflow.keras.optimizers.SGD(learning_rate=0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
with tensorflow.device('/GPU:0'):
  history = model.fit(training_data,
                        steps_per_epoch = int(24268/batch_size),
                        epochs = 50,
                        validation_data = valid_data,
                        validation_steps = int(8348/batch_size))

history.model.save('/content/drive/My Drive/model/data_contrast_lr=1.5e-4_[15:-1]_0.55_full.h5')
test_data = valid_gen.flow_from_directory(directory=test_direction,
                                            target_size=(128,128),
                                            batch_size=batch_size,
                                            class_mode='binary',
                                          shuffle = False)
from sklearn.metrics import classification_report, confusion_matrix
Y_pred = model.predict(test_data)
y_pred = np.argmax(Y_pred, axis=1)
print(test_data.classes[100:500])
print(y_pred[100:500])

print(Y_pred.shape)
print('Confusion Matrix')
print(confusion_matrix(test_data.classes, y_pred))
print('Classification Report')
print(classification_report(test_data.classes, y_pred))