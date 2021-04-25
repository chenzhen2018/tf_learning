# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/29
# @Author  : chenzhen
# @File    : keras_save_and_load_test.py

import os
import h5py
import numpy as np
import tensorflow as tf

from libs.load_keras_dataset import load_mnist

K = tf.keras.backend

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

"""
Save and Load Model of Keras

"""

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
    initial_learning_rate = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.1, beta_2=0.2, amsgrad=True),
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.metrics.SparseCategoricalAccuracy()])
    return model


# 以saved model的形式保存整个模型
# model = create_model()
# model.summary()
# model.fit(train_images, train_labels, epochs=10)
# model.save("./saved_model/chen/training_1")
#
# reconstructed_model = tf.keras.models.load_model('./saved_model/chen/training_1')
# res = np.testing.assert_allclose(
#     model.predict(test_images), reconstructed_model.predict(test_images)
# )
# print(res)
#
# print("---------------------")
# reconstructed_model.fit(test_images, test_labels)


# 以HDF5的形式保存模型
# model = create_model()
# model.summary()
# model.fit(train_images, train_labels, epochs=10)
# model.save("./saved_model/chen/training_2/model.h5")
#
# reconstructed_model = tf.keras.models.load_model("./saved_model/chen/training_2/model.h5")
# res = np.testing.assert_allclose(
#     model.predict(test_images), reconstructed_model.predict(test_images)
# )
# print(res)
#
# print("---------------------")
# reconstructed_model.fit(test_images, test_labels)


# 在训练过程中保存模型
# model = create_model()
# model.summary()
#
# checkpoint_path = "./saved_model/chen/training_5/cp"
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=False,
#                                                  verbose=1)
# tf.keras.callbacks.LearningRateScheduler
# model.fit(train_images,
#           train_labels, epochs=10,
#           validation_data=(test_images, test_labels),
#           callbacks=[cp_callback])
# tf.keras.callbacks.BaseLogger
class my_callback(tf.keras.callbacks.Callback):
    def __int__(self):
        super(my_callback, self).__int__()
    def on_epoch_begin(self, epoch, logs=None):
        # if not hasattr(self.model.optimizer, 'lr'):
        #     raise ValueError('Optimizer must have a "lr" attribute.')
        lr = K.get_value(self.model.optimizer.lr)
        print()
        # K.set_value(self.model.optimizer.lr, K.get_value(lr))
        # # print("===Training lr: {}".format(lr))
        # print("===== {}, {}".format(epoch, lr))

    # def on_train_epoch_begin(self, batch, logs=None):
    #     if not hasattr(self.model.optimizer, 'lr'):
    #         raise ValueError('Optimizer must have a "lr" attribute.')
    #     lr = float(K.eval(self.model.optimizer.lr))
    #     print("===Training lr: {}".format(lr))

model = create_model()
model.fit(train_images, train_labels,
          epochs=10,
          # validation_data=(test_images, test_labels),
          callbacks=[my_callback()])
model.save('./saved_model/chen/save_1/mymodel')
model.save_weights('./saved_model/chen/save_2/mymodel')

model.save('./saved_model/chen/save_3/mymodel.h5')
model.save_weights('./saved_model/chen/save_4/mymodel.h5')

# from tensorflow.python.framework import ops
#
# class LearningRateScheduler(tf.keras.callbacks.Callback):
#
#   def __init__(self, schedule, verbose=0):
#     super(LearningRateScheduler, self).__init__()
#     self.schedule = schedule
#     self.verbose = verbose
#
#   def on_epoch_begin(self, epoch, logs=None):
#     # if not hasattr(self.model.optimizer, 'lr'):
#     #   raise ValueError('Optimizer must have a "lr" attribute.')
#     lr = float(K.get_value(self.model.optimizer.lr))
#     lr = self.schedule(epoch, lr)
#     K.set_value(self.model.optimizer.lr, K.get_value(lr))
#   # print('\nEpoch %05d: LearningRateScheduler reducing learning '
#   #       'rate to %s.' % (epoch + 1, lr))
#     print("------test, {}".format(lr))
#   # def on_epoch_end(self, epoch, logs=None):
#   #   logs = logs or {}
#   #   logs['lr'] = K.get_value(self.model.optimizer.lr)
#
# def scheduler(epoch, lr):
#     if epoch < 10:
#         return lr
#     else:
#         return lr * tf.math.exp(-0.1)
# model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
# model.compile(tf.keras.optimizers.SGD(), loss='mse')
# print(round(model.optimizer.lr.numpy(), 5))
#
# callback = LearningRateScheduler(scheduler)
# history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
#                     epochs=15, callbacks=[callback], verbose=0)
# print(round(model.optimizer.lr.numpy(), 5))
