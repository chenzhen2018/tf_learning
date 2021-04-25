# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/26
# @Author  : chenzhen
# @File    : plot_predictions.py

import numpy as np
import matplotlib.pyplot as plt

"""

:plot_image(predictions_arr, true_label, img, class_names), plot predictions
:plot_value_array(predictions_array, true_label, num_classes), bar

"""


def plot_image(predictions_array, true_label, img, class_names):
    """

    :param predictions_array: a prediction
    :param true_label:  a label
    :param img:  an image
    :param class_names: 
    :return:
    """
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(
        class_names[predicted_label],
        100*np.max(predictions_array),
        class_names[true_label]
    ), color=color)


def plot_value_array(predictions_array, true_label, num_classes=10):
    """

    :param predictions_array: a prediction
    :param true_label: a label
    :param num_classes:
    :return:
    """
    plt.grid(False)
    plt.xticks(range(num_classes))
    plt.yticks([])

    thisplot = plt.bar(range(num_classes), predictions_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def plot_nums_imgs(num_rows, num_cols, predictions, test_labels, test_images, class_names):
    """
    plot num_rows*num_cols results
    :param num_rows:
    :param num_cols:
    :param predictions:  all predictions
    :param test_labels: all test labels
    :param test_images: all test images
    :param class_names:
    :return:
    """
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(predictions[i], test_labels[i], test_images[i], class_names)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(predictions[i], test_labels[i])
    plt.tight_layout()
