# -*- coding: utf-8 -*-  
# author: ytq, chenzhen@lognshine.com
# =================================================

import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from PIL import Image

"""
高级教程-图像-使用TF Hub进行迁移学习
https://tensorflow.google.cn/tutorials/images/transfer_learning_with_hub
"""

##########################
# Download the classifier
##########################
classifier_model ="https://hub.tensorflow.google.cn/google/tf2-preview/mobilenet_v2/classification/4"
IMAGE_SHAPE = (224, 224)
classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE + (3,))
])
print(classifier.summary())

# test: Run the model on a single image
grace_hopper = tf.keras.utils.get_file('grace_hopper.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
grace_hopper = np.array(grace_hopper)/255.0
result = classifier.predict(grace_hopper[np.newaxis, ...])
print(result.shape)
predicted_class = np.argmax(result[0], axis=-1)
print(predicted_class)

# Decode the predictions
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())
print("Predictions: {}".format(imagenet_labels[predicted_class]))

##########################
# Simple trnsfer learning
##########################
# dataset
data_root = tf.keras.utils.get_file(
  'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
   untar=True)

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
ds_train = tf.keras.preprocessing.image_dataset_from_directory(str(data_root),
                                                               validation_split=0.2,
                                                               subset='training',
                                                               seed=123,
                                                               image_size=IMG_SIZE,
                                                               batch_size=BATCH_SIZE)
class_names = ds_train.class_names
# normalize
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
ds_train = ds_train.map(lambda x, y: (normalization_layer(x), y))
ds_train = ds_train.cache().prefetch(tf.data.experimental.AUTOTUNE)
# test:
for image_batch, labels_batch in ds_train:
    print(image_batch.shape, labels_batch.shape)
    break
# Run the classifier on a batch of images
# result_batch = classifier.predict(ds_train)
# print(result_batch)

# Download the headless model
feature_extractor_model = "https://hub.tensorflow.google.cn/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor_layer = hub.KerasLayer(feature_extractor_model,
                                         input_shape=(224, 224, 3),
                                         trainable=False)
feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)

# Attach a classification head
num_classes = len(class_names)
model = tf.keras.Sequential([
    feature_extractor_layer,
    tf.keras.layers.Dense(num_classes)
])
print(model.summary())
# test
predictions = model(image_batch)
print(predictions.shape)

# Train the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['acc'])


# Callbacks
class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []
    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()
batch_stats_callback = CollectBatchStats()
history = model.fit(ds_train, epochs=2,
                    callbacks=[batch_stats_callback])
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(batch_stats_callback.batch_losses)

plt.subplot(2, 1, 2)
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(batch_stats_callback.batch_acc)

plt.savefig('./test/tfhub_transfer_learning.png')

######################
# Export your model
######################
t = time.time()
export_path = './test/tfhub/saved_models/{}'.format(int(t))
model.save(export_path)

reloaded_model = tf.keras.models.load_model(export_path)
result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded_model.predict(image_batch)

print(abs(reloaded_result_batch - result_batch).max())

