# -*- coding: utf-8 -*-  
# author: ytq, chenzhen@lognshine.com
# =================================================

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

mpl.use("TkAgg")
tf.random.set_seed(123)
sns_colors = sns.color_palette('colorblind')
pd.set_option('display.max_columns', None)

"""
理解梯度提升树
https://tensorflow.google.cn/tutorials/estimator/boosted_trees_model_understanding
"""

# =============================
# ======== 加载数据 ===========
# =============================
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# ==========================================================
# ==== Create Feature columns, Input fn, Train =============
# ==========================================================
fc = tf.feature_column
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

def one_hot_cat_column(feature_name, vocab):
  return fc.indicator_column(
      fc.categorical_column_with_vocabulary_list(feature_name,
                                                 vocab))
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  # 需要使用独热编码（one-hot-encoding）处理非数值特征。
  vocabulary = dftrain[feature_name].unique()
  feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(fc.numeric_column(feature_name,
                                           dtype=tf.float32))

NUM_EXAMPLES = len(y_train)
def make_input_fn(X, y, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((X.to_dict(orient='list'), y))
        if shuffle:
            dataset = dataset.shuffle(NUM_EXAMPLES)
        dataset = dataset.repeat(n_epochs).batch(NUM_EXAMPLES)
        return dataset
    return input_fn
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, n_epochs=1)

params = {
  'n_trees': 50,
  'max_depth': 3,
  'n_batches_per_layer': 1,
  # 为了得到 DFCs，请设置 center_bias = True。这将强制
  # 模型在使用特征（例如：回归中训练集标签的均值，分类中使
  # 用交叉熵损失函数时的对数几率）前做一个初始预测。
  'center_bias': True
}
est = tf.estimator.BoostedTreesClassifier(feature_columns, **params)
# 训练模型。
est.train(train_input_fn, max_steps=100)
# 评估。
results = est.evaluate(eval_input_fn)
res = pd.Series(results).to_frame()
print(res)

pred_dicts = list(est.experimental_predict_with_explanations(eval_input_fn))
labels = y_eval.values
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
df_dfc = pd.DataFrame([pred['dfc'] for pred in pred_dicts])
print(df_dfc.describe().T)
# print(pred_dicts)


bias = pred_dicts[0]['bias']
dfc_prob = df_dfc.sum(axis=1) + bias
print(np.testing.assert_almost_equal(dfc_prob.values, probs.values))






