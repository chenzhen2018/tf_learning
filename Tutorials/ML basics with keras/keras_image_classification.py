# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/26
# @Author  : chenzhen
# @File    : keras_image_classification.py

import os
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from libs.fashion_mnist import load_dataset
from libs import plot_classification_predictions
mpl.use("Agg")

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

"""

"""

# Load dataset
path_fashion_mnist = '/home/chenz/data/fashion_mnist'
(train_images, train_labels), (test_images, test_labels) = load_dataset.load_data(path_fashion_mnist)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print("train images: {}, {}, train_labels: {}, {}, \ntest_images: {}, {} test_labels: {}, {}".format(
    train_images.shape, type(train_images), train_labels.shape, type(train_labels),
    test_images.shape, type(test_images), test_labels.shape, type(test_labels)
))
print("train_labels: {}, test_labels: {}".format(set(np.reshape(train_labels, (-1,))),
                                                 set(np.reshape(test_labels, (-1,)))))
# display an image
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.savefig("./imgs/1.png")

# Preprocess operations
train_images = train_images / 255.0
test_images = test_images / 255.0

# display 25 images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.savefig('./imgs/2.png')

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile the model: setting loss function, optimizer, metrics
model.compile(optimizer='adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Model Training
model.fit(train_images, train_labels, epochs=10)

# Evaluate model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy: ", test_acc)

# Predict--add softmax layer
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)




i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1,2,1)
plot_classification_predictions.plot_image(predictions[i], test_labels[i], test_images[i], class_names)
plt.subplot(1, 2, 2)
plot_classification_predictions.plot_value_array(predictions[i], test_labels[i])
plt.savefig('./imgs/3.png')


plot_classification_predictions.plot_nums_imgs(num_rows=5,
                                               num_cols=3,
                                               predictions=predictions,
                                               test_labels=test_labels,
                                               test_images=test_images,
                                               class_names=class_names)
plt.savefig('./imgs/4.png')

plt.close()












