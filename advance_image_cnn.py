# -*- coding: utf-8 -*-  
# author: ytq, chenzhen@lognshine.com
# =================================================

import tensorflow as tf
import matplotlib.pyplot as plt

"""
高级教程-图像-CNN
https://tensorflow.google.cn/tutorials/images/cnn
"""


##############################
# Download and Prepare dataset
##############################
(train_images, train_labels), (test_images, test_lables) = tf.keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0  # normalize
print("Train data: {}, Train labels: {}".format(train_images.shape, train_labels.shape))
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

###################
# Build the Model
###################
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])
print(model.summary())
# model compile
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# model fit
history = model.fit(train_images, train_labels,
                    epochs=10,
                    validation_data=(test_images, test_lables))

###################
# Plot Training Process
####################
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.savefig('./test/training_process.png')

#####################
# Test dataset evaluate
######################
test_loss, test_acc = model.evaluate(test_images, test_lables, verbose=2)

print("Test Acc: {}, Test Loss: {}".format(test_acc, test_loss))

