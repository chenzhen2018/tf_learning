# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/28
# @Author  : chenzhen
# @File    : data_tfrecord_tfexample.py

import numpy as np
import tensorflow as tf


# ===========================
# ===== Feature =============
# ==========================

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

print(_bytes_feature(b'test_string'))
print(_bytes_feature(u'test_bytest'.encode('utf-8')))
print(_float_feature(np.exp(1)))
print(_int64_feature(True))
print(_int64_feature(1))

feature = _float_feature(np.exp(1))
print(feature.SerializeToString())

# ==============================
# ====== tf.Example ============
# ==============================
n_observations = int(1e4)
feature0 = np.random.choice([False, True], n_observations)
feature1 = np.random.randint(0, 5, n_observations)
strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]
feature3 = np.random.randn(n_observations)

def serialize_example(feature0, feature1, feature2, feature3):
    # Create a dict mapping the feature name to the tf.Exampel-compatible data type
    feature = {
        'feature0': _int64_feature(feature0),
        'feature1': _int64_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _float_feature(feature3)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

serialized_example = serialize_example(False, 4, b'goat', 0.98)
print(serialized_example, type(serialized_example))

example_proto = tf.train.Example.FromString(serialized_example)
print(example_proto, type(example_proto))


# ========================================
# === TFRecord Write and Read ============
# ========================================
ds_1 = tf.data.Dataset.from_tensor_slices(feature0)
print(ds_1, type(ds_1))

ds_2 = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))
print(ds_2, type(ds_2))

for f0, f1, f2, f3 in ds_2.take(1):
    print(f0, f0.numpy())
    print(f1, f1.numpy())
    print(f2, f2.numpy())
    print(f3, f3.numpy())

def tf_serialize_example(f0, f1, f2, f3):
    tf_string = tf.py_function(
        serialize_example,
        (f0, f1, f2, f3),
        tf.string
    )
    return tf.reshape(tf_string, ())
print(tf_serialize_example(f0, f1, f2, f3))

serialized_features_dataset = ds_2.map(tf_serialize_example)
print(serialized_features_dataset)


def generator():
    for features in ds_2:
        yield serialize_example(*features)

serialized_features_dataset = tf.data.Dataset.from_generator(generator, output_types=tf.string, output_shapes=())
print(serialized_features_dataset)


filename = './test/test.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)

filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
print(raw_dataset)

for raw_record in raw_dataset.take(10):
    print(repr(raw_record))

# Create a description of the features
feature_description = {
    'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0)
}
def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)

parsed_dataset = raw_dataset.map(_parse_function)
print(parsed_dataset)

for parsed_record in parsed_dataset.take(10):
    print(repr(parsed_record))


with tf.io.TFRecordWriter(filename) as writer:
    for i in range(n_observations):
        example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
        writer.write(example)

filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
print(raw_dataset)
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

    print(example)


# ===============================================================
print('======================================================')

cat_path = '/home/chenz/data/320px-Felis_catus-cat_on_snow.jpg'
bridge_path = '/home/chenz/data/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg'

image_labels = {
    cat_path: 0,
    bridge_path: 1
}

# This is an example, just using the cat image
image_string = open(cat_path, 'rb').read()
label = image_labels[cat_path]

# Create a dict with features that may be revelant
def image_example(image_string, label):
    image_shape = tf.image.decode_jpeg(image_string).shape
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

for line in str(image_example(image_string, label)).split('\n')[:15]:
    print(line)
print('...')

record_file = './test/images.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
    for filename, label in image_labels.items():
        image_string = open(filename, 'rb').read()
        tf_example = image_example(image_string, label)
        writer.write(tf_example.SerializeToString())

raw_image_dataset = tf.data.TFRecordDataset('./test/images/tfrecords')
# Create a dict describing the features
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string)
}
def _parse_image_function(example_proto):
    return (tf.io.parse_single_example(example_proto, image_feature_description))
parsed_image_dataset = raw_dataset.map(_parse_image_function)
print(parsed_image_dataset)

for image_features in parsed_image_dataset:
    image_raw = image_features['image_raw'].numpy()
    print(image_raw)