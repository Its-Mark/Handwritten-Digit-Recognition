# Train with convolutional neural networks (CNN)

# Import libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
# Divide training and testing datasets training = 60,000 & testing = 10,000
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()


# Pre-process & Normalize data before CNN
xTrain = tf.keras.utils.normalize(xTrain, axis = 1)
xTest = tf.keras.utils.normalize(xTest, axis = 1)
plt.imshow(xTrain[0], cmap=plt.cm.binary)

x_trainr = np.array(xTrain).reshape(-1, 28, 28, 1)
x_testr = np.array(xTest).reshape(-1, 28, 28, 1)

# Build our CNN model
model = tf.keras.models.Sequential()

# Convolution Layer 1
model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=x_trainr.shape[1]))
model.add(tf.keras.layers.Activation(relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Convolution Layer 2
model.add(tf.keras.layers.Conv2D(64, (3, 3)))
model.add(tf.keras.layers.Activation(relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Convolution Layer 3
model.add(tf.keras.layers.Conv2D(64, (3, 3)))
model.add(tf.keras.layers.Activation(relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Fully Connected Layer 1
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Activation3(relu))

# Fully Connected Layer 2
model.add(tf.keras.layers.Dense(32))
model.add(tf.keras.layers.Activation(relu))

# Fully Connected Layer 3
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation(softmax))

# Compile the model
model.compile(loss=sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
epochs = 10
# Train the model
history = model.fit(x_trainr, yTrain, epochs=epochs, validation_split=0.3)

# Evaluate the model
loss, accuracy = model.evaluate(x_testr, yTest)
print(loss , loss)
print(accuracy , accuracy)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()