
#%pip install tensorflow

import os, shutil
import random
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D,Dropout
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.layers import MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


plt.style.use('ggplot')

print(os.listdir(".")) #Bu kod ayni dizindeki dosyalari listeler. iki nokta ust dizini listeler

DIR = os.listdir('./chest_xray/chest_xray')  

print(DIR)  

labels = ['PNEUMONIA','NORMAL']
img_size = 128
def get_data(data_dir):
    data=[]
    for label in labels:
#         train/PNEUMONIA
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data, dtype=object) 


train = "./chest_xray/chest_xray/train"
test = "./chest_xray/chest_xray/test"
val = "./chest_xray/chest_xray/val"

pneumonia = os.listdir("./chest_xray/chest_xray/train/PNEUMONIA")
penomina_dir = "./chest_xray/chest_xray/train/PNEUMONIA"


IMG_HEIGHT = 128
IMG_WIDTH = 128

BATCH_SIZE = 32


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    brightness_range=(1.2, 1.5),
    horizontal_flip=True
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)


train_data = train_datagen.flow_from_directory(
    train,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary',
    batch_size=BATCH_SIZE,
    color_mode='grayscale' 
)

val_data = train_datagen.flow_from_directory(
    val,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary',
    batch_size=BATCH_SIZE,
    color_mode='grayscale' 
)

test_data = train_datagen.flow_from_directory(
    test,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary',
    batch_size=BATCH_SIZE,
    color_mode='grayscale' 
)

# Model
first_model=Sequential()
first_model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1))) #sondaki 1 renksiz resim oldugu icin. Renkli olsaydi 3 olacakti
first_model.add(MaxPool2D(2,2))
first_model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
first_model.add(MaxPool2D(2,2))
first_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
first_model.add(MaxPool2D(2,2))
first_model.add(Flatten())
first_model.add(Dense(256, activation='relu'))
first_model.add(Dense(128, activation='relu'))
first_model.add(Dense(1, activation='sigmoid')) # Eger 2 den fazla class varsa softmax kullanilir

first_model.summary()


# compile model
first_model.compile(loss='binary_crossentropy',
                    optimizer=Adam(learning_rate=0.001),
                    metrics=['accuracy'])


history = first_model.fit(train_data,
                          epochs=15,
                          verbose=1,
                          validation_data=val_data
                          )

loss, accuracy = first_model.evaluate(test_data)
print("Test: accuracy = %f  ;  loss = %f " % (accuracy, loss))


# Predicting the test data using the predict function
predictions = first_model.predict(test_data)

# Retrieving the class labels with the highest probability
predicted_classes = np.argmax(predictions, axis=1)


# Loading the Model
first_model.save("cnn_model.keras")

# Processes an image file and prepares it in a format suitable for the model.
labels = ["NORMAL", "PNEUMONIA"]
def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  
    new_array = cv2.resize(img_array, (IMG_HEIGHT, IMG_WIDTH))  
    return new_array.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1) 

# extra pneumonia photo from google
prediction = first_model.predict([prepare("lung_300.jpg")])
print(labels[int(prediction[0])])