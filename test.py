import csv
import io
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
from google.colab import files
train = pd.read_csv(io.StringIO(uploadTrain['asteroid_training.csv'].decode('utf-8'))).replace('\n','', regex=True).values
test  = pd.read_csv(io.StringIO(uploadedTest['asteroid_testing.csv'].decode('utf-8'))).replace('\n','', regex=True).values


trainX = train[:, 3:] #respective reflectance values
trainY = train[:,2] #label (composition type) of each asteroids in training set

testX = test[:, 3:] #respective reflectance values
testY = test[:,2] #label (composition type) of each asteroids in testing set



#Supervised Neural Network model. 3 Layers.
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(46, activation='relu')) #relu layer
model.add(tf.keras.layers.Dense(46, activation='relu')) #relu layer
model.add(tf.keras.layers.Dense(4, activation='softmax')) #softmax layer for prediciton
model.compile(tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#actual training and validation  
history = model.fit(trainX, trainY, epochs=200, batch_size=46,
          validation_data=(testX, testY))



# basic graph of accuracy and loss.

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print('PREDICTION TEST ON CERES ASTEROID')
def matchOutputToClassCeres():
  predictions = model.predict(testX)
  print(predictions[0])
  predictionNum =np.argmax(predictions[0])
  print(predictionNum)
  print('PREDICITOIN TEST')
  if (predictionNum == 0):
    print('I think Ceres belong to C class')
  elif (predictionNum == 1):
    print('I think Ceres belong to E class')
  elif (predictionNum == 2):
    print('I think Ceres belong to X class')
  elif (predictionNum == 3):
    print('I think Ceres belong to S class')
  print('TRUE VALUE')
  print('Ceres belong to C class')
matchOutputToClassCeres()
