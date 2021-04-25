#!/usr/bin/env python3
# -*- coding: utf-8 -*-  
# author: chenzhen

"""
在Keras中使用RNN
"""

import numpy as np
import tensorflow as tf


model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=1000, output_dim=64))
model.add(tf.keras.layers.LSTM(128, return_sequences=True))
model.add(tf.keras.layers.Dense(10))

model.summary()