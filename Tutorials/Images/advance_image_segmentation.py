# -*- coding: utf-8 -*-  
# author: ytq, chenzhen@lognshine.com
# =================================================

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow_ex


"""
高级教程-图像-图像分割
https://tensorflow.google.cn/tutorials/images/segmentation
"""

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
print(dataset)
print(dataset.keys())
print(info)
print(type(dataset), type(info))


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

TRAIN_LENGTH = info.splits['train'].num_examples
print(TRAIN_LENGTH)
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

# Get Train/Test Dataset
ds_train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = dataset['test'].map(load_image_test)
print(type(ds_train))
print(ds_train)

ds_train = ds_train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)


def display(display_list, num=0):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.savefig('./test/seg_{}.png'.format(str(num)))
    plt.close()


for image, mask in ds_test.take(1):
    sample_image, sample_mask = image, mask
print(sample_image.shape)
display([sample_image[0, :], sample_mask[0, :]], num=0)


####################
# Create the Model
####################
OUTPUT_CHANNELS = 3
base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
print(base_model.summary())
# test
sample_pred = base_model(sample_image)
print(sample_pred.shape)

# 使用这些层的激活设置
layer_names = [
    'block_1_expand_relu',  # 64x64
    'block_3_expand_relu',  # 64x64
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]
# 创建特征提取模型
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
down_stack.trainable = False
print(down_stack.summary())