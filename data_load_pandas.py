# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/28
# @Author  : chenzhen
# @File    : data_load_pandas.py


import pandas as pd
import tensorflow as tf


data_path = '/home/chenz/data/applied-dl/heart.csv'

df_data = pd.read_csv(data_path)
print(df_data.head())
print(df_data.dtypes)

df_data['thal'] = pd.Categorical(df_data['thal'])
print(df_data.head())
df_data['thal'] = df_data.thal.cat.codes
print(df_data.head())

# construct dataset
target = df_data.pop('target')
dataset = tf.data.Dataset.from_tensor_slices((df_data.values, target.values))
for feat, targ in dataset.take(5):
    print("Features: {}, Target: {}".format(feat, targ))

train_dataset = dataset.shuffle(len(df_data)).batch(1)

def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
model = get_compiled_model()
model.fit(train_dataset, epochs=15)

print('------------------------------------------------')
inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df_data.keys()}
x = tf.stack(list(inputs.values()), axis=-1)
x = tf.keras.layers.Dense(10, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model_func = tf.keras.Model(inputs, output)
model_func.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

dict_slices = tf.data.Dataset.from_tensor_slices((df_data.to_dict('list'), target.values)).batch(16)

for dict_slice in dict_slices.take(1):
    print(dict_slice)

model_func.fit(dict_slices, epochs=15)