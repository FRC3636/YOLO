from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2


data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0



imgTest = cv2.imread("Capture.png", 0)
resizedImg = cv2.resize(imgTest, (28, 28))
invertedImg = cv2.bitwise_not(resizedImg)


#cv2.imshow("imgTest", imgTest)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()])

#myTestArray = {invertedImg}

x_train[len(x_train)-1] = invertedImg
predictions = probability_model.predict(x_train)
#predictions = probability_model.predict(myTestArray[0])

print(np.argmax(predictions[len(x_train)-1]))


plt.figure()
plt.imshow(x_train[len(x_train)-1])
plt.colorbar()
plt.show()



#plt.figure()
#plt.imshow(invertedImg)
#plt.colorbar()
#plt.show()


cv2.waitKey(0)

