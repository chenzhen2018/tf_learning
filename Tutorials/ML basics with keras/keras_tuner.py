# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/19
# @Author  : chenzhen
# @File    : keras_tuner.py


from libs.load_keras_dataset import load_mnist

import tensorflow as tf
import kerastuner as kt
from libs.load_keras_dataset import load_fashion_mnist

path_fashion_mnist = '/home/chenz/data/fashion_mnist'

(x_train, y_train), (x_test, y_test) = load_fashion_mnist(path_fashion_mnist)
print("[INFO] x_train: {}, y_train: {}".format(x_train.shape, y_train.shape))
print("[INFO] x_test: {}, y_test: {}".format(x_test.shape, y_test.shape))


# Define the model
def model_builder(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    # Tune the number of units in the first Dense layer
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(tf.keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(tf.keras.layers.Dense(10))

    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# Instantiate the tuner and preform hypertuning
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')
tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hps)
print(tuner.results_summary())

# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

