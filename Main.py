from __future__ import absolute_import, division, print_function, unicode_literals

import cv2
import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow import keras



fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

img = cv2.imread("Capture.png", 0)
img = cv2.resize(img, (28,28))
img = cv2.bitwise_not(img)




plt.figure()
plt.imshow(img)
plt.colorbar()
plt.grid(False)
plt.show()

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()])
#predictions = probability_model.predict(test_images)

img = (np.expand_dims(img, 0))

singlePrediction = probability_model.predict(img)

print(class_names[np.argmax(singlePrediction[0])])

