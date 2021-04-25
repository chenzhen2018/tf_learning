# -*- coding: utf-8 -*-  
# author: ytq, chenzhen@lognshine.com
# =================================================

import os
import h5py
import tensorflow as tf

from libs.load_keras_dataset import load_mnist

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

"""
Save and Load Model of Keras

"""

# Load dataset
mnist_path = '/home/chenz/data/mnist/mnist.npz'
(x_train, y_train), (x_test, y_test) = load_mnist(data_path=mnist_path)
print("[INFO] x_train: {}, y_train: {}, x_test: {}, y_test: {}".format(
    x_train.shape, y_train.shape, x_test.shape, y_test.shape
))
train_labels = y_train[:1000]
test_labels = y_test[:1000]

train_images = x_train[:1000].reshape(-1, 28*28) / 255.0
test_images = x_test[:1000].reshape(-1, 28*28) / 255.0

print("[INFO] train_images: {}, train_labels: {}, test_images: {}, test_labels: {}".format(
    train_images.shape, train_labels.shape, test_images.shape, test_labels.shape
))

# Build Model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.metrics.SparseCategoricalAccuracy()])
    return model


# ==================================ModelCheckpoint==================================
# model = create_model()
# model.summary()
#
# checkpoint_path = 'saved_model/training_1/cp.ckpt'
# checkpoint_dir = os.path.dirname(checkpoint_path)
#
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)
# model.fit(train_images,
#           train_labels,
#           epochs=10,
#           validation_data=(test_images, test_labels),
#           callbacks=[cp_callback])
#
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
#
# model.load_weights(checkpoint_path)
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))
# ==================================ModelCheckpoint==================================


# ==================================ModelCheckpoint-epoch==================================
# checkpoint_path = "saved_model/training_2/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path,
#     verbose=1,
#     save_weights_only=True,
#     period=5
# )
# model = create_model()
# model.save_weights(checkpoint_path.format(epoch=0))
#
# model.fit(train_images,
#           train_labels,
#           epochs=50,
#           callbacks=[cp_callback],
#           validation_data=(test_images, test_labels),
#           verbose=0)
#
# latest = tf.train.latest_checkpoint(checkpoint_dir)
# print(latest)
#
# model = create_model()
# model.load_weights(latest)
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))
# ==================================ModelCheckpoint-epoch==================================

# ==================================Manually save weights==================================
# model = create_model()
# model.summary()
# model.fit(train_images,
#           train_labels,
#           epochs=50,
#           validation_data=(test_images, test_labels),
#           verbose=0)
#
# # Save the weights
# model.save_weights('./saved_model/checkpoints/my_checkpoint')
# # Create a new model instance
# model = create_model()
# # Restore the weights
# model.load_weights('./saved_model/checkpoints/my_checkpoint')
# # Evaluate the model
# loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))
# ==================================Manually save weights==================================

# ==================================Save the entire model(ckpt)==================================
# model = create_model()
# model.fit(train_images, train_labels, epochs=5)
# model.save('saved_model/my_model')
#
# # load
# new_model = tf.keras.models.load_model('saved_model/my_model')
# new_model.summary()
#
# # Evaluate the restored model
# loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
# print('Restored model, accuracy: {:5.2f}%'.format(100*acc))
# print(new_model.predict(test_images).shape)
# ==================================Save the entire model(ckpt)==================================

# ==================================Save the entire model(h5)==================================
# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)
model.save('./saved_model/my_model.h5')

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('./saved_model/my_model.h5')
# Show the model architecture
new_model.summary()

loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

# ==================================Save the entire model(h5)==================================
