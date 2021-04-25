# -*- coding: utf-8 -*-  
# author: ytq, chenzhen@lognshine.com
# =================================================

import tensorflow as tf
import pandas as pd

"""
https://tensorflow.google.cn/tutorials/estimator/premade
"""

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
print('-------------- training dataset ------------------')
print(train.head())
# 获取标签
train_y = train.pop('Species')
test_y = test.pop('Species')
print(train.head())
print(train.keys(), type(train.keys()))


def input_fn(features, labels, training=True, batch_size=256):
    """An input function for training or evaluating"""
    # 将输入转换为数据集。
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # 如果在训练模式下混淆并重复数据。
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

# 定义特征列 feature columns
# 特征列描述了如何使用输入。
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)

# 实例化Estimator
# 构建一个拥有两个隐层，隐藏节点分别为 30 和 10 的深度神经网络。
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # 隐层所含结点数量分别为 30 和 10.
    hidden_units=[30, 10],
    # 模型必须从三个类别中做出选择。
    n_classes=3)
# 训练模型。
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)

# 评估模型
eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))
print(eval_result)
print("\nTest set accuracy: {accuracy:0.3f}\n".format(**eval_result))

# Inference
# 由模型生成预测
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}
def input_fn(features, batch_size=256):
    """An input function for prediction."""
    # 将输入转换为无标签数据集。
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)
predictions = classifier.predict(
    input_fn=lambda: input_fn(predict_x))

print(type(predictions))
for pred_dict, expec in zip(predictions, expected):
    print(type(pred_dict), pred_dict)
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
        SPECIES[class_id], 100 * probability, expec))