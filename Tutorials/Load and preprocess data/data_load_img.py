# -*- coding: utf-8 -*-  
# author: ytq, chenzhen@lognshine.com
# =================================================

import pathlib
import tensorflow as tf

tf.keras.utils.get_file

data_root_orig = '/home/chenz/data/'

data_root = pathlib.Path(data_root_orig)
print(data_root, type(data_root))

