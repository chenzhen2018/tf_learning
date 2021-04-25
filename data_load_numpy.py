# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/28
# @Author  : chenzhen
# @File    : data_load_numpy.py

import tensorflow as tf
from libs import load_keras_dataset

data_path = '/home/chenz/data/mnist/mnist.npz'
(x_train, y_train), (x_test, y_test) = load_keras_dataset.load_mnist(data_path)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100
ds_train = ds_train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
ds_test = ds_test.batch(BATCH_SIZE)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
model.fit(ds_train, epochs=10)

model.evaluate(ds_test)