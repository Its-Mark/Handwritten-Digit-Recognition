# RNN USING LSTM LAYERS
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, LSTM, Dropout
import cv2 as cv
import matplotlib.pyplot as plt

# import the mnist dataset that contains images of handwritten digits
(xTrain, yTrain), (xTest, yTest) = keras.datasets.mnist.load_data()

# normalize training and testing data to be btwn 0 - 1
xTrain = tf.keras.utils.normalize(xTrain, axis=1)
xTest = tf.keras.utils.normalize(xTest, axis=1)


model = tf.keras.models.Sequential()
# add LSTM layers to model followed by a dropout layer as my "output" layer
model.add(LSTM(128, input_shape=(xTrain.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
# compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# train out model with the training sets
history = model.fit(xTrain, yTrain, epochs=3, validation_data=(xTest, yTest))
# get the accuracy and loss of model when tested with testing sets
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

print("Model has an accuracy of", acc)
print("Model has a loss of ", loss)
model.summary()
# Visualize Loss and Accuracy charts
epocsR = range(3)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epocsR, acc, label='Training Accuracy')
plt.plot(epocsR, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epocsR, loss, label='Training Loss')
plt.plot(epocsR, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# loop to read in images drawn into a 28x28px png file.
# assumes file names are 1.png, 2.png ..... x.png
for x in range(10):
    img = cv.imread(f'{x}.png')[:, :, 0]
    img = np.invert(np.array([img]))
    # normalize the data
    img = tf.keras.utils.normalize(img, axis=1)
    # use our model to predict what was on the image
    prediction = model.predict(img)
    print(f'Image {x} is similar to: {np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()



# ##################################################################################
# # BELOW FOR RNN MODEL#
# # scale down training/testing data to be btwn 0 - 1
# xTrain = tf.keras.utils.normalize(xTrain, axis=1)
# xTest = tf.keras.utils.normalize(xTest, axis=1)
#
# # build our rnn model
# model = tf.keras.models.Sequential()
# # add our layers and LSTM layers
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
# # output layer
# model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
#
# # compile model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# # train the rnn model
# model.fit(xTrain, yTrain, epochs=3)
#
# loss, accuracy = model.evaluate(xTest, yTest)
# print(accuracy)
# print(loss)
#
# model.save('digits.model')
