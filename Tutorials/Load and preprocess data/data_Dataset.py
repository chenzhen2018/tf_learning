# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/28
# @Author  : chenzhen
# @File    : data_Dataset.py


import numpy as np
import tensorflow as tf


dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))
print(dataset1, type(dataset1), dataset1.element_spec, dataset1.element_spec.value_type)

dataset2 = tf.data.Dataset.from_tensor_slices((tf.random.uniform([4]),
                                               tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))
print(dataset2, type(dataset2), dataset2.element_spec, dataset2.element_spec[0], dataset2.element_spec[0].value_type)

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3, dataset3.element_spec)

print('---------------------------------')
def count(stop):
    i = 0
    while i < stop:
        yield i
        i += 1

ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.int32, output_shapes=())

for count_batch in ds_counter.repeat().batch(10).take(10):
    print(count_batch.numpy())

print('-----------------------')
def gen_serise():
    i = 0
    while True:
        size = np.random.randint(0, 10)
        yield i, np.random.normal(size=(size, ))
        i += 1
for i, serise in gen_serise():
    print("i: {}".format(str(serise)))
    if i > 5:
        break

ds_series = tf.data.Dataset.from_generator(
    gen_serise,
    output_types=(tf.int32, tf.float32),
    output_shapes=((), (None,))
)
print(ds_series)

ds_series_batch = ds_series.padded_batch(10)

ids, sequence_batch = next(iter(ds_series_batch))
print(ids.numpy())
print()
print(sequence_batch.numpy())

print('----------------------------')
flowers = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)
img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20)
images, labels = next(img_gen.flow_from_directory(flowers))
print(images.dtype, images.shape)
print(labels.dtype, labels.shape)