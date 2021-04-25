# -*- coding: utf-8 -*-  
# author: ytq, chenzhen@lognshine.com
# =================================================

import tensorflow as tf


def vgg():
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', name='l1_conv1')(inputs)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', name='l1_conv2')(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', name='l2_conv1')(x)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', name='l2_conv2')(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', name='l3_conv1')(x)
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', name='l3_conv2')(x)
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', name='l3_conv3')(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', name='l4_conv1')(x)
    x = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', name='l4_conv2')(x)
    x = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', name='l4_conv3')(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', name='l5_conv1')(x)
    x = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', name='l5_conv2')(x)
    x = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', name='l5_conv3')(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Flatten(name='l6_flatten')(x)
    x = tf.keras.layers.Dense(4096, activation='relu', name='l6_conv1')(x)
    x = tf.keras.layers.Dense(1000, activation='relu', name='l6_conv2')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return model




if __name__ == '__main__':

    model = vgg()
    print(model.summary())
