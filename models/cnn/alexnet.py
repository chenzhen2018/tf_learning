# -*- coding: utf-8 -*-  
# author: ytq, chenzhen@lognshine.com
# =================================================


import tensorflow as tf

def alexnet():
    inputs = tf.keras.layers.Input(shape=(227, 227, 3), name='inputs')

    # Layer 1
    x = tf.keras.layers.Conv2D(96, 11, strides=(4, 4), activation='relu', name='l1_conv')(inputs)
    x = tf.keras.layers.MaxPool2D(3, strides=2, name='l1_maxpool')(x)
    x = tf.keras.layers.BatchNormalization(name='l1_bn')(x)

    # Layer 2
    x = tf.keras.layers.Conv2D(256, kernel_size=5, padding='same', activation='relu', name='l2_conv')(x)
    x = tf.keras.layers.MaxPool2D(3, strides=2, name='l2_maxpool')(x)
    x = tf.keras.layers.BatchNormalization(name='l2_bn')(x)

    # Layer 3
    x = tf.keras.layers.Conv2D(384, kernel_size=3, padding='same', activation='relu', name='l3_conv')(x)
    x = tf.keras.layers.BatchNormalization(name='l3_bn')(x)

    # Layer 4
    x = tf.keras.layers.Conv2D(384, kernel_size=3, padding='same', activation='relu', name='14_conv')(x)
    x = tf.keras.layers.BatchNormalization(name='l4_bn')(x)

    # Layer 5
    x = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu', name='l5_conv4')(x)
    x = tf.keras.layers.MaxPool2D(3, strides=2, name='l5_maxpool')(x)
    x = tf.keras.layers.BatchNormalization(name='l5_bn')(x)

    # Layer 6
    x = tf.keras.layers.Flatten(name='l6_flatten')(x)
    x = tf.keras.layers.Dense(4096, activation='relu', name='l6_fc')(x)
    x = tf.keras.layers.Dropout(rate=0.2, name='l6_dropout')(x)

    # Layer 7
    x = tf.keras.layers.Dense(4096, activation='relu', name='l7_fc')(x)
    x = tf.keras.layers.Dropout(rate=0.2, name='l7_dropout')(x)

    # Layer 8
    x = tf.keras.layers.Dense(1000, activation='softmax', name='l8_fc_output')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return model


if __name__ == '__main__':
    model = alexnet()
    print(model.summary())
