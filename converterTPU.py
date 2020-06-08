import tensorflow as tf
import os
tf.enable_eager_execution()
from tensorflow import keras
print(tf.version.VERSION)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split


#Importing Data from CSV file
data=pd.read_csv("fer2013.csv")
emotion=data['emotion']
for i in range(0,35886):
    if emotion[i]==0:
        emotion[i]=0
    elif emotion[i]==1:
        emotion[i]=0
    elif emotion[i]==2:
        emotion[i]=0
    elif emotion[i]==3:
        emotion[i]=1
    elif emotion[i]==4:
        emotion[i]=0
    elif emotion[i]==5:
        emotion[i]=1
    elif emotion[i]==6:
        emotion[i]=2
data.to_csv("fer2013_modi.csv")
data=pd.read_csv("fer2013_modi.csv")
labels=data.iloc[:,[1]].values
pixels=data['pixels']

#Facial Expressions
#Expressions={0:"Angry",1:"Disgust",2:"Fear",3:"Happy",4:"Sad",5:"Surprise",6:"Neutral"}
Expressions={0:"Positive",1:"Negative",2:"Neutral"}
from keras.utils import to_categorical 
labels = to_categorical(labels,len(Expressions))

#converting pixels to Gray Scale images of 48X48 
images = np.array([np.fromstring(pixel, dtype=int, sep=" ")for pixel in pixels])
images=images/255.0
images = images.reshape(images.shape[0],48,48,1).astype('float32')

plt.imshow(images[0][:,:,0])
Expressions[labels[0][0]]

#splitting data into training and test data
train_images,test_images,train_labels,test_labels = train_test_split(images,labels,test_size=0.2,random_state=0)

train_labels

def create_model(classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32,kernel_size=(2,2),strides=(1,1),activation='relu',input_shape=(48,48,1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(filters=64,kernel_size=(2,2),strides=(1,1),activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(1,1)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(filters=128,kernel_size=(2,2),strides=(1,1),activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(1,1)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(filters=256,kernel_size=(2,2),strides=(1,1),activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(1,1)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Dense(classes,activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

classes = 3
model = create_model(classes)

#train the CNN 
model.fit(train_images,train_labels,batch_size=80,epochs=30,verbose=2)

label_pred=model.predict(test_images)
label_pred=np.argmax(label_pred,axis = 1)

#making confusion matrix
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

test_labels=np.argmax(test_labels,axis=1)
# Compute confusion matrix
class_names=Expressions

#Save the model
filename='facial_model.h5'
model.save(filename,overwrite=True)
dt_dir = '/home/tungna/Dropbox/fer2013'

from os import listdir
from os.path import isfile, join
def representative_data_gen():
    dataset_list = tf.data.Dataset.list_files(dt_dir + '/*/*/*')
    for i in range(10):
        image = next(iter(dataset_list))
        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image, channels=1)
        image = tf.image.resize(image, [48, 48])
        image = tf.cast(image, tf.float32)/255.0
        image = tf.expand_dims(image, 0)
        yield [image]

#Convert the Keras file to TfLite:
converter = tf.lite.TFLiteConverter.from_keras_model_file(filename)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_data_gen
tflite_model = converter.convert()

with open('facial_model.tflite','wb') as f:
   f.write(tflite_model)


