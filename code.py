
'''Data preprocessing: 
Converting categorical columns into numeric. Coulmn Total Charges had some null values. 
Filling null values with 0.
'''

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import metrics

df = pd.read_csv ('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df_dummies = pd.get_dummies(df, columns = ['gender', 'Partner', 'Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','Churn'], drop_first = True)

df_dummies["TotalCharges"] = pd.to_numeric(df_dummies.TotalCharges, errors='coerce')
df_dummies["TotalCharges"] = df_dummies['TotalCharges'].fillna(0)
df_dummies.drop("customerID", axis= 1, inplace= True)


x = df_dummies[df_dummies.columns[0:30]]
y = df_dummies["Churn_Yes"]
x_train, x_test, y_train, y_test = train_test_split(  x, y , train_size = 0.7,random_state= 67)

#Building neural network using keras

from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
import keras
from keras.utils import to_categorical
from keras import models
from keras import layers

'''As the Y variable is categorical - binary, hence using sigmoid as activation function
and binary cross entropy as loss function.''' 

model = keras.Sequential([
 keras.layers.Dense(64, activation=tf.nn.relu,
 input_shape=(x_train.shape[1],)),
 keras.layers.Dense(64, activation=tf.nn.relu),
 keras.layers.Dense(1 , activation = 'sigmoid')
 ])
optimizer = tf.train.RMSPropOptimizer(0.001)
model.compile(
optimizer = "adam",
 loss = "binary_crossentropy",
 metrics = ["accuracy"])
model.summary()

OUTPUT:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_4 (Dense)              (None, 64)                1984      
_________________________________________________________________
dense_5 (Dense)              (None, 64)                4160      
_________________________________________________________________
dense_6 (Dense)              (None, 1)                 65        
=================================================================
Total params: 6,209
Trainable params: 6,209
Non-trainable params: 0

# Fit the model:
customer_churn = model.fit(
 x_train, y_train,
 epochs= 30,
 validation_data = (x_test, y_test))
 
 OUTPUT

Train on 4930 samples, validate on 2113 samples
Epoch 1/30
4930/4930 [==============================] - 1s 111us/step - loss: 2.7358 - acc: 0.7306 - val_loss: 2.1183 - val_acc: 0.7482
Epoch 2/30
4930/4930 [==============================] - 0s 63us/step - loss: 3.1063 - acc: 0.7475 - val_loss: 3.0480 - val_acc: 0.7553
Epoch 3/30
4930/4930 [==============================] - 0s 67us/step - loss: 2.6211 - acc: 0.7473 - val_loss: 3.0240 - val_acc: 0.7525
Epoch 4/30
4930/4930 [==============================] - 0s 93us/step - loss: 3.0148 - acc: 0.7564 - val_loss: 3.0427 - val_acc: 0.7525
Epoch 5/30
4930/4930 [==============================] - 0s 67us/step - loss: 2.8219 - acc: 0.7590 - val_loss: 2.8248 - val_acc: 0.7619
Epoch 6/30
4930/4930 [==============================] - 0s 91us/step - loss: 3.3592 - acc: 0.7454 - val_loss: 3.9140 - val_acc: 0.7321
Epoch 7/30
4930/4930 [==============================] - 0s 91us/step - loss: 3.2206 - acc: 0.7578 - val_loss: 3.0355 - val_acc: 0.7567
Epoch 8/30
4930/4930 [==============================] - 0s 91us/step - loss: 2.7702 - acc: 0.7633 - val_loss: 3.8127 - val_acc: 0.7321
Epoch 9/30
4930/4930 [==============================] - 0s 93us/step - loss: 3.2071 - acc: 0.7611 - val_loss: 3.0362 - val_acc: 0.7572
Epoch 10/30
4930/4930 [==============================] - 0s 90us/step - loss: 2.7151 - acc: 0.7736 - val_loss: 2.4751 - val_acc: 0.7378
Epoch 11/30
4930/4930 [==============================] - 0s 89us/step - loss: 2.8972 - acc: 0.7448 - val_loss: 3.2197 - val_acc: 0.7553
Epoch 12/30
4930/4930 [==============================] - 0s 90us/step - loss: 2.9803 - acc: 0.7639 - val_loss: 2.8003 - val_acc: 0.7582
Epoch 13/30
4930/4930 [==============================] - 0s 87us/step - loss: 2.6329 - acc: 0.7604 - val_loss: 3.2056 - val_acc: 0.7331
Epoch 14/30
4930/4930 [==============================] - 0s 89us/step - loss: 2.7713 - acc: 0.7495 - val_loss: 3.1832 - val_acc: 0.7563
Epoch 15/30
4930/4930 [==============================] - 0s 89us/step - loss: 2.9616 - acc: 0.7665 - val_loss: 2.8253 - val_acc: 0.7591
Epoch 16/30
4930/4930 [==============================] - 0s 88us/step - loss: 2.3107 - acc: 0.7753 - val_loss: 2.8073 - val_acc: 0.7321
Epoch 17/30
4930/4930 [==============================] - 0s 90us/step - loss: 2.2482 - acc: 0.7515 - val_loss: 2.0103 - val_acc: 0.7667
Epoch 18/30
4930/4930 [==============================] - 0s 62us/step - loss: 2.4303 - acc: 0.7560 - val_loss: 2.8818 - val_acc: 0.7610
Epoch 19/30
4930/4930 [==============================] - 0s 67us/step - loss: 2.7165 - acc: 0.7631 - val_loss: 2.8579 - val_acc: 0.7615
Epoch 20/30
4930/4930 [==============================] - 0s 63us/step - loss: 2.4549 - acc: 0.7588 - val_loss: 2.9885 - val_acc: 0.7321
Epoch 21/30
4930/4930 [==============================] - 0s 65us/step - loss: 2.3021 - acc: 0.7531 - val_loss: 1.8278 - val_acc: 0.7534
Epoch 22/30
4930/4930 [==============================] - 0s 64us/step - loss: 2.4797 - acc: 0.7572 - val_loss: 2.1079 - val_acc: 0.7601
Epoch 23/30
4930/4930 [==============================] - 0s 64us/step - loss: 2.0583 - acc: 0.7515 - val_loss: 2.1914 - val_acc: 0.7743
Epoch 24/30
4930/4930 [==============================] - 0s 62us/step - loss: 1.9808 - acc: 0.7558 - val_loss: 2.0117 - val_acc: 0.7705
Epoch 25/30
4930/4930 [==============================] - 0s 62us/step - loss: 1.4055 - acc: 0.7552 - val_loss: 1.3689 - val_acc: 0.7894
Epoch 26/30
4930/4930 [==============================] - 0s 65us/step - loss: 1.4931 - acc: 0.7570 - val_loss: 1.0085 - val_acc: 0.7922
Epoch 27/30
4930/4930 [==============================] - 0s 62us/step - loss: 1.5635 - acc: 0.7584 - val_loss: 1.4600 - val_acc: 0.7790
Epoch 28/30
4930/4930 [==============================] - 0s 64us/step - loss: 0.9435 - acc: 0.7680 - val_loss: 0.7012 - val_acc: 0.7866
Epoch 29/30
4930/4930 [==============================] - 0s 64us/step - loss: 0.9100 - acc: 0.7704 - val_loss: 0.8102 - val_acc: 0.7832
Epoch 30/30
4930/4930 [==============================] - 0s 62us/step - loss: 0.7676 - acc: 0.7613 - val_loss: 0.5048 - val_acc: 0.8031


#Accuracy of the model:

accuracy = model.evaluate(x_train, y_train)
print('Train accuracy:', accuracy)
_, accuracy = model.evaluate(x_test, y_test)
print('Test accuracy:', accuracy)

OUTPUT:
4930/4930 [==============================] - 0s 30us/step
Train accuracy: 0.8018255578093306
2113/2113 [==============================] - 0s 31us/step
Test accuracy: 0.8031235210601041

#Visualize the results

import matplotlib.pyplot as plt
def plot_history(customer_churn):
 plt.figure()
 plt.xlabel('Epoch')
 plt.ylabel('Accuracy [1000$]')
 plt.plot(customer_churn.epoch, np.array(customer_churn.history['acc']),
 label='Train Loss')
 plt.plot(customer_churn.epoch, np.array(customer_churn.history['val_acc']),
 label = 'Val loss')
 plt.legend()
 plt.ylim([0.5, 1])
plot_history(customer_churn)


