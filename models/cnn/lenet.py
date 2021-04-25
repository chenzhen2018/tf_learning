# -*- coding: utf-8 -*-  
# author: ytq, chenzhen@lognshine.com
# =================================================


import tensorflow as tf


def lenet():
    inputs = tf.keras.Input(shape=(None, 32, 32))
    # conv1
    x = tf.keras.layers.Conv2D(6, 5, activation='relu')(inputs)
    # pool1
    x = tf.keras.layers.MaxPool2D()(x)
    # conv2
    x = tf.keras.layers.Conv2D(16, 5, activation='relu')(x)
    # pool2
    x = tf.keras.layers.MaxPool2D()(x)
    # conv3
    x = tf.keras.layers.Conv2D(120, 5, activation='relu')(x)

    # fc
    x = tf.keras.layers.Dense(120, activation='relu')(x)
    x = tf.keras.layers.Dense(10, activation='relu')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x)

    return model


if __name__ == '__main__':
    model = lenet()
    print(model.summary())