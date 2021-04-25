# -*- coding: utf-8 -*-  
# author: ytq, chenzhen@lognshine.com
# =================================================

import tensorflow as tf



def inceptionV1():
    # Input
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    # Conv
    x = tf.keras.layers.ZeroPadding2D(padding=1)(inputs)
    x = tf.keras.layers.Conv2D(64, 7, strides=2, activation='relu', name='pre_conv1')(x)
    # Max pooling
    x = tf.keras.layers.ZeroPadding2D(padding=2)(x)
    x = tf.keras.layers.MaxPool2D(3, strides=2, name='pre_pool1')(x)
    # Conv
    x = tf.keras.layers.Conv2D(192, 3, strides=1, padding='same', activation='relu', name='pre_conv2')(x)
    # Max Pooling
    x = tf.keras.layers.MaxPool2D(name='pre_pool2')(x)

    # Inception 3a
    branch1x1 = tf.keras.layers.Conv2D(64, 1, padding='same', activation='relu', name='inception_3a_branch1x1')(x)
    branch3x3 = tf.keras.layers.Conv2D(96, 1, padding='same', activation='relu', name='inception_3a_branch3x3_reduce')(x)
    branch3x3 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', name='inception_3a_branch3x3')(branch3x3)
    branch5x5 = tf.keras.layers.Conv2D(16, 1, padding='same', activation='relu', name='inception_3a_branch5x5_reduce')(x)
    branch5x5 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', name='inception_3a_branch5x5')(branch5x5)
    branch_pool = tf.keras.layers.MaxPool2D(3, 1, padding='same', name='inception_3a_branchpool')(x)
    branch_pool = tf.keras.layers.Conv2D(32, 1, padding='same', activation='relu', name='inception_3a_branchpool_reduce')(branch_pool)
    x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1, name='inception_3a_concate')

    # Inception 3b
    branch1x1 = tf.keras.layers.Conv2D(128, 1, padding='same', activation='relu', name='inception_3b_branch1x1')(x)
    branch3x3 = tf.keras.layers.Conv2D(128, 1, padding='same', activation='relu', name='inception_3b_branch3x3_reduce')(
        x)
    branch3x3 = tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu', name='inception_3b_branch3x3')(
        branch3x3)
    branch5x5 = tf.keras.layers.Conv2D(32, 1, padding='same', activation='relu', name='inception_3b_branch5x5_reduce')(
        x)
    branch5x5 = tf.keras.layers.Conv2D(96, 3, padding='same', activation='relu', name='inception_3b_branch5x5')(
        branch5x5)
    branch_pool = tf.keras.layers.MaxPool2D(3, 1, padding='same', name='inception_3b_branchpool')(x)
    branch_pool = tf.keras.layers.Conv2D(64, 1, padding='same', activation='relu',
                                         name='inception_3b_branchpool_reduce')(branch_pool)
    x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1,
                                    name='inception_3b_concate')

    # Max Pooling
    x = tf.keras.layers.MaxPool2D(3, 2, padding='same', name='max_pool_1')(x)

    # Inception 4a
    branch1x1 = tf.keras.layers.Conv2D(192, 1, padding='same', activation='relu', name='inception_4a_branch1x1')(x)
    branch3x3 = tf.keras.layers.Conv2D(96, 1, padding='same', activation='relu', name='inception_4a_branch3x3_reduce')(
        x)
    branch3x3 = tf.keras.layers.Conv2D(208, 3, padding='same', activation='relu', name='inception_4a_branch3x3')(
        branch3x3)
    branch5x5 = tf.keras.layers.Conv2D(16, 1, padding='same', activation='relu', name='inception_4a_branch5x5_reduce')(
        x)
    branch5x5 = tf.keras.layers.Conv2D(48, 3, padding='same', activation='relu', name='inception_4a_branch5x5')(
        branch5x5)
    branch_pool = tf.keras.layers.MaxPool2D(3, 1, padding='same', name='inception_4a_branchpool')(x)
    branch_pool = tf.keras.layers.Conv2D(64, 1, padding='same', activation='relu',
                                         name='inception_4a_branchpool_reduce')(branch_pool)
    x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1,
                                    name='inception_4a_concate')

    # Inception 4b
    branch1x1 = tf.keras.layers.Conv2D(160, 1, padding='same', activation='relu', name='inception_4b_branch1x1')(x)
    branch3x3 = tf.keras.layers.Conv2D(112, 1, padding='same', activation='relu', name='inception_4b_branch3x3_reduce')(
        x)
    branch3x3 = tf.keras.layers.Conv2D(224, 3, padding='same', activation='relu', name='inception_4b_branch3x3')(
        branch3x3)
    branch5x5 = tf.keras.layers.Conv2D(24, 1, padding='same', activation='relu', name='inception_4b_branch5x5_reduce')(
        x)
    branch5x5 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', name='inception_4b_branch5x5')(
        branch5x5)
    branch_pool = tf.keras.layers.MaxPool2D(3, 1, padding='same', name='inception_4b_branchpool')(x)
    branch_pool = tf.keras.layers.Conv2D(64, 1, padding='same', activation='relu',
                                         name='inception_4b_branchpool_reduce')(branch_pool)
    x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1,
                                    name='inception_4b_concate')

    # Inception 4c
    branch1x1 = tf.keras.layers.Conv2D(128, 1, padding='same', activation='relu', name='inception_4c_branch1x1')(x)
    branch3x3 = tf.keras.layers.Conv2D(128, 1, padding='same', activation='relu', name='inception_4c_branch3x3_reduce')(
        x)
    branch3x3 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', name='inception_4c_branch3x3')(
        branch3x3)
    branch5x5 = tf.keras.layers.Conv2D(24, 1, padding='same', activation='relu', name='inception_4c_branch5x5_reduce')(
        x)
    branch5x5 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', name='inception_4c_branch5x5')(
        branch5x5)
    branch_pool = tf.keras.layers.MaxPool2D(3, 1, padding='same', name='inception_4c_branchpool')(x)
    branch_pool = tf.keras.layers.Conv2D(64, 1, padding='same', activation='relu',
                                         name='inception_4c_branchpool_reduce')(branch_pool)
    x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1,
                                    name='inception_4c_concate')

    # Inception 4d
    branch1x1 = tf.keras.layers.Conv2D(112, 1, padding='same', activation='relu', name='inception_4d_branch1x1')(x)
    branch3x3 = tf.keras.layers.Conv2D(144, 1, padding='same', activation='relu', name='inception_4d_branch3x3_reduce')(
        x)
    branch3x3 = tf.keras.layers.Conv2D(288, 3, padding='same', activation='relu', name='inception_4d_branch3x3')(
        branch3x3)
    branch5x5 = tf.keras.layers.Conv2D(32, 1, padding='same', activation='relu', name='inception_4d_branch5x5_reduce')(
        x)
    branch5x5 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', name='inception_4d_branch5x5')(
        branch5x5)
    branch_pool = tf.keras.layers.MaxPool2D(3, 1, padding='same', name='inception_4d_branchpool')(x)
    branch_pool = tf.keras.layers.Conv2D(64, 1, padding='same', activation='relu',
                                         name='inception_4d_branchpool_reduce')(branch_pool)
    x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1,
                                    name='inception_4d_concate')

    # Inception 4e
    branch1x1 = tf.keras.layers.Conv2D(256, 1, padding='same', activation='relu', name='inception_4e_branch1x1')(x)
    branch3x3 = tf.keras.layers.Conv2D(160, 1, padding='same', activation='relu', name='inception_4e_branch3x3_reduce')(
        x)
    branch3x3 = tf.keras.layers.Conv2D(320, 3, padding='same', activation='relu', name='inception_4e_branch3x3')(
        branch3x3)
    branch5x5 = tf.keras.layers.Conv2D(32, 1, padding='same', activation='relu', name='inception_4e_branch5x5_reduce')(
        x)
    branch5x5 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', name='inception_4e_branch5x5')(
        branch5x5)
    branch_pool = tf.keras.layers.MaxPool2D(3, 1, padding='same', name='inception_4e_branchpool')(x)
    branch_pool = tf.keras.layers.Conv2D(128, 1, padding='same', activation='relu',
                                         name='inception_4e_branchpool_reduce')(branch_pool)
    x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1,
                                    name='inception_4e_concate')

    # Max Pooling
    x = tf.keras.layers.MaxPool2D(3, 2, padding='same', name='max_pool_2')(x)

    # Inception 5a
    branch1x1 = tf.keras.layers.Conv2D(256, 1, padding='same', activation='relu', name='inception_5a_branch1x1')(x)
    branch3x3 = tf.keras.layers.Conv2D(160, 1, padding='same', activation='relu', name='inception_5a_branch3x3_reduce')(
        x)
    branch3x3 = tf.keras.layers.Conv2D(320, 3, padding='same', activation='relu', name='inception_5a_branch3x3')(
        branch3x3)
    branch5x5 = tf.keras.layers.Conv2D(32, 1, padding='same', activation='relu', name='inception_5a_branch5x5_reduce')(
        x)
    branch5x5 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', name='inception_5a_branch5x5')(
        branch5x5)
    branch_pool = tf.keras.layers.MaxPool2D(3, 1, padding='same', name='inception_5a_branchpool')(x)
    branch_pool = tf.keras.layers.Conv2D(128, 1, padding='same', activation='relu',
                                         name='inception_5a_branchpool_reduce')(branch_pool)
    x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1,
                                    name='inception_5a_concate')

    # Inception 5b
    branch1x1 = tf.keras.layers.Conv2D(384, 1, padding='same', activation='relu', name='inception_5b_branch1x1')(x)
    branch3x3 = tf.keras.layers.Conv2D(192, 1, padding='same', activation='relu', name='inception_5b_branch3x3_reduce')(
        x)
    branch3x3 = tf.keras.layers.Conv2D(384, 3, padding='same', activation='relu', name='inception_5b_branch3x3')(
        branch3x3)
    branch5x5 = tf.keras.layers.Conv2D(48, 1, padding='same', activation='relu', name='inception_5b_branch5x5_reduce')(
        x)
    branch5x5 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', name='inception_5b_branch5x5')(
        branch5x5)
    branch_pool = tf.keras.layers.MaxPool2D(3, 1, padding='same', name='inception_5b_branchpool')(x)
    branch_pool = tf.keras.layers.Conv2D(128, 1, padding='same', activation='relu',
                                         name='inception_5b_branchpool_reduce')(branch_pool)
    x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1,
                                    name='inception_5b_concate')

    # Average Pooling
    x = tf.keras.layers.AveragePooling2D(7, strides=1, name='avg_pool')(x)

    # Dropout
    x = tf.keras.layers.Dropout(0.4, name='dropout')(x)
    # Linear & Softmax
    x = tf.keras.layers.Dense(1000, name='linear')(x)
    x = tf.keras.layers.Dense(1000, activation='softmax', name='output')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x)

    return model


if __name__ == '__main__':
    model = inceptionV1()
    print(model.summary())