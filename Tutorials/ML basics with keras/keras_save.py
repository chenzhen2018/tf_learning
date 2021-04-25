# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/7
# @Author  : chenzhen
# @File    : keras_save.py


import tensorflow as tf
from libs.load_keras_dataset import load_mnist


# Load dataset
mnist_path = '/home/chenz/data/mnist/mnist.npz'
(x_train, y_train), (x_test, y_test) = load_mnist(data_path=mnist_path)
print("[INFO] x_train: {}, y_train: {}, x_test: {}, y_test: {}".format(
    x_train.shape, y_train.shape, x_test.shape, y_test.shape
))
train_labels = y_train[:1000]
test_labels = y_test[:1000]
train_images = x_train[:1000].reshape(-1, 28*28) / 255.0
test_images = x_test[:1000].reshape(-1, 28*28) / 255.0

print("[INFO] train_images: {}, train_labels: {}, test_images: {}, test_labels: {}".format(
    train_images.shape, train_labels.shape, test_images.shape, test_labels.shape
))

# Build Model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.1, beta_2=0.2, amsgrad=True),
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.metrics.SparseCategoricalAccuracy()])
    return model


# ========= method 1: checkpoint =====================
# model = create_model()
#
# checkpoint_path = "./saved_model/save_and_load/cp_test_1/cp.ckpt"
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=False,
#                                                  verbose=1)
# model.fit(train_images, train_labels,
#           epochs=3,
#           validation_data=(test_images, test_labels),
#           callbacks=[cp_callback])

# model.save_weights()
# model.save(filepath="./saved_model/save_and_load/save_test/test_7",
#            save_format='tf')
# model.save_weights(filepath="./saved_model/save_and_load/save_test/weights/weights_9",
#                    save_format='tf')


# ==============load-=======================
# model_path = './saved_model/save_and_load/save_test/test_5/'
#
# model = tf.keras.models.load_model(model_path)
# model.summary()

model = create_model()
model.load_weights("./saved_model/save_and_load/save_test/weights/weights_1")
model.summary()

