# -*- coding: utf-8 -*-  
# author: ytq, chenzhen@lognshine.com
# =================================================

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


"""
高级教程-图像-数据增强
https://tensorflow.google.cn/tutorials/images/data_augmentation
"""

######################
# Download a dataset
######################
(ds_train, ds_val, ds_test), metadata = tfds.load('tf_flowers',
                                                  split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
                                                  with_info=True, as_supervised=True)
num_classes = metadata.features['label'].num_classes
print(num_classes)
class_names = metadata.features['label'].int2str
print(class_names)
# test: ds_train
image, label = next(iter(ds_train))
print(image, type(image))
print(label, type(label))
print(class_names[label])

#################################
# Use Keras preprocessing layers
#################################
IMG_SIZE = 180
resize_and_rescale = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
])
# test: preprocessing
result = resize_and_rescale(image)
print("Min and max pixel values: ", result.numpy().min(), result.numpy().max())

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# Create the model
model = tf.keras.Sequential([
    resize_and_rescale,
    data_augmentation,
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    # Rest of your model
])
# or: aug_ds = train_ds.map(lambda x, y: (resize_and_rescale(x, training=True), y))

# Apply the preprocessing layers to the datasets
BATCH_SIZE = 32
def prepare(ds, shuffle=False, augment=False):
    # or: aug_ds = ds.map(lambda x, y: (resize_and_rescale(x, training=True), y))
    if shuffle:
        ds = ds.shuffle(1000)
    # Batch all datasets
    ds = ds.batch(BATCH_SIZE)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

train_ds = prepare(ds_train, shuffle=True, augment=True)
val_ds = prepare(ds_val)
test_ds = prepare(ds_test)

# Train a model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
epochs=5
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Custom data augmentation
def random_invert_img(x, p=0.5):
  if  tf.random.uniform([]) < p:
    x = (255-x)
  else:
    x
  return x

def random_invert(factor=0.5):
  return tf.keras.layers.Lambda(lambda x: random_invert_img(x, factor))

random_invert = random_invert()

class RandomInvert(tf.keras.layers.Layer):
  def __init__(self, factor=0.5, **kwargs):
    super().__init__(**kwargs)
    self.factor = factor

  def call(self, x):
    return random_invert_img(x)


#######################
# Using tf.image
#######################
(train_ds, val_ds, test_ds), metadata = tfds.load('tf_flowers',
                                                  split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
                                                  with_info=True,
                                                  as_supervised=True)
image, label = next(iter(train_ds))
# Data augmentation
flipped = tf.image.flip_left_right(image)
grayscaled = tf.image.rgb_to_grayscale(image)
saturated = tf.image.adjust_saturation(image, 3)
bright = tf.image.adjust_brightness(image, 0.4)
cropped = tf.image.central_crop(image, central_fraction=0.5)
rotated = tf.image.rot90(image)
