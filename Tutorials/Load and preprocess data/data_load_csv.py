# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/26
# @Author  : chenzhen
# @File    : data_load_csv.py

import functools
import numpy as np
import pandas as pd
import tensorflow as tf
np.set_printoptions(precision=3, suppress=True)


train_file_path = '/home/chenz/data/titanic/train.csv'
test_file_path = '/home/chenz/data/titanic/eval.csv'

pd_train_data = pd.read_csv(train_file_path)
print(pd_train_data.head())

LABEL_COLUM = 'survived'
LABELS = [0, 1]

def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name=LABEL_COLUM,
        na_value='?',
        num_epochs=1,
        ignore_errors=True
    )
    return dataset
raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

print('-------------------------')
examples, labels = next(iter(raw_train_data))
print(examples)
print('-------------------------')
print(labels)


CATEGORIES = {
    'sex': ['male', 'female'],
    'class' : ['First', 'Second', 'Third'],
    'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone' : ['y', 'n']
}
categorical_columns = []
for feature, vocab in CATEGORIES.items():
  cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
  categorical_columns.append(tf.feature_column.indicator_column(cat_col))
print(categorical_columns)

def process_continuous_data(mean, data):
    data = tf.cast(data, tf.float32) * 1/ (2*mean)
    return tf.reshape(data, [-1,1])
MEANS = {
    'age' : 29.631308,
    'n_siblings_spouses' : 0.545455,
    'parch' : 0.379585,
    'fare' : 34.385399
}
numerical_columns = []
for feature in MEANS.keys():
    num_col = tf.feature_column.numeric_column(feature, normalizer_fn=functools.partial(process_continuous_data, MEANS[feature]))
    numerical_columns.append(num_col)
print(numerical_columns)
