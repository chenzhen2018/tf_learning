# -*- coding: utf-8 -*-  
# author: ytq, chenzhen@lognshine.com
# =================================================

import os
import matplotlib.pyplot as plt


def plot_history(history, names, max_val=10, is_val=False, save_path=None):
    """

    :param history:
    :param losses:
    :param acc:
    :return:
    """
    print("All keys in history: {}".format(history.history.keys()))
    num_axes = len(names)
    rows = num_axes // 2 + 1

    hist = history.history
    epoch = history.epoch
    plt.figure()
    for i, name in enumerate(names):
        plt.subplot(rows, rows, i+1)
        plt.xlabel("Epoch")
        plt.ylabel(name)
        plt.plot(epoch, hist[name], label=name)
        if is_val:
            plt.plot(epoch, hist['val_'+name], label='val_'+name)
        plt.ylim([0, max_val])
        plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)

    # save
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
    plt.close()