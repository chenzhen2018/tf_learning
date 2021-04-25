# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/27
# @Author  : chenzhen
# @File    : keras_overfiting_underfiting.py

import os
import shutil
import pathlib
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("Agg")


"""

"""

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

logdir = pathlib.Path(tempfile.mktemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)


FEATURES = 28
data_path = '/home/chenz/data/HIGGS/HIGGS.csv.gz'
ds = tf.data.experimental.CsvDataset(data_path, [float(),]*(FEATURES+1), compression_type="GZIP")
print(type(ds))


def pack_row(*row):
    label = row[0]
    features = tf.stack(row[1:], 1)
    return features, label
packed_ds = ds.batch(10000).map(pack_row).unbatch()
print(type(packed_ds))

for features, label in packed_ds.batch(1000).take(1):
    print(features[0], label[0])
    plt.hist(features.numpy().flatten(), bins=101)
    plt.savefig('./imgs/9.png')

N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE

validate_ds = packed_ds.take(N_VALIDATION).cache()
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()

validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)
print(train_ds)

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=STEPS_PER_EPOCH*1000,
    decay_rate=1,
    staircase=False
)


def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)

step = np.linspace(0, 100000)
print(type(step), step.shape, step)
lr = lr_schedule(step)
plt.figure(figsize=(8, 6))
plt.plot(step/STEPS_PER_EPOCH, lr)
plt.ylim([0, max(plt.ylim())])
plt.xlabel("EPOCH")
_ = plt.ylabel("Learning Rate")
plt.savefig('./imgs/10.png')


def get_callbacks(name):
    return [
        tfdocs.modeling.EpochDots(),
        tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
        tf.keras.callbacks.TensorBoard(logdir/name),
    ]

def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
    if optimizer is None:
        optimizer = get_optimizer()
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[
                      tf.keras.losses.BinaryCrossentropy(
                          from_logits=True, name='binary_crossentropy'
                      ),
                      'accuracy'
                  ])

    model.summary()
    history = model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=max_epochs,
        validation_data=validate_ds,
        callbacks = get_callbacks(name),
        verbose=0
    )
    return history

tiny_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    tf.keras.layers.Dense(1)
])
small_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    tf.keras.layers.Dense(16, activation='elu'),
    tf.keras.layers.Dense(1)
])
medium_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='elu', input_shape=(FEATURES,)),
    tf.keras.layers.Dense(64, activation='elu'),
    tf.keras.layers.Dense(64, activation='elu'),
    tf.keras.layers.Dense(1)
])
large_model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    tf.keras.layers.Dense(512, activation='elu'),
    tf.keras.layers.Dense(512, activation='elu'),
    tf.keras.layers.Dense(512, activation='elu'),
    tf.keras.layers.Dense(1)
])
l2_model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='elu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=(FEATURES,)),
    tf.keras.layers.Dense(512, activation='elu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dense(512, activation='elu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dense(512, activation='elu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dense(1)
])
dropout_model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='elu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='elu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='elu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])
combined_model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='elu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001), input_shape=(FEATURES,)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='elu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='elu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='elu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])

plotter = tfdocs.plots.HistoryPlotter(metric='binary_crossentropy', smoothing_std=10)


# size_histories = {}
# size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')
# size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small')
# size_histories['Medium'] = compile_and_fit(medium_model, 'sizes/Medium')
# size_histories['Large'] = compile_and_fit(large_model, 'sizes/Large')
#


regularizer_histories = {}
regularizer_histories['Tiny'] = compile_and_fit(tiny_model, 'regularizers/Tiny')
regularizer_histories['l2'] = compile_and_fit(l2_model, 'regularizers/l2')
regularizer_histories['dropout'] = compile_and_fit(dropout_model, 'regularizers/Dropout')
regularizer_histories['combined'] = compile_and_fit(combined_model, 'regularizers/Combined')


plotter.plot(regularizer_histories)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
plt.ylim([0.5, 0.7])
plt.xlabel("Epochs [Log Scale]")
plt.savefig("./imgs/11.png")