# -*- coding: utf-8 -*-  
# author: ytq, chenzhen@lognshine.com
# =================================================

import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

from libs.data_windowing import WindowGenerator

mpl.use("Agg")


data_path = '/home/chenz/data/jena_climate_2009_2016.csv'

df_data = pd.read_csv(data_path)
df_data = df_data[5::6]
date_time = pd.to_datetime(df_data.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
print(df_data.head())
print(date_time.head())
print(df_data.describe().transpose())

# ===================================================
# =========== Inspect and cleanup ===================
# ===================================================
# Wind velocity
wv = df_data['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0

max_wv = df_data['max. wv (m/s)']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0
print(df_data.describe().transpose())

# ===================================================
# =========== Feature engineering ===================
# ===================================================
# wind
wv = df_data.pop('wv (m/s)')
max_wv = df_data.pop('max. wv (m/s)')
# Convert to radians.
wd_rad = df_data.pop('wd (deg)')*np.pi / 180
# # Calculate the wind x and y components.
df_data['Wx'] = wv*np.cos(wd_rad)
df_data['Wy'] = wv*np.sin(wd_rad)
# Calculate the max wind x and y components.
df_data['max Wx'] = max_wv*np.cos(wd_rad)
df_data['max Wy'] = max_wv*np.sin(wd_rad)

# Time
timestamp_s = date_time.map(datetime.datetime.timestamp)
day = 24*60*60
year = (365.2425)*day
df_data['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df_data['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df_data['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df_data['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
print(df_data.head())

# ========================================
# =========== Split the data =============
# ========================================
column_indices = {name: i for i, name in enumerate(df_data.columns)}
n = len(df_data)
train_df = df_data[0:int(n*0.7)]
val_df = df_data[int(n*0.7):int(n*0.9)]
test_df = df_data[int(n*0.9):]
num_features = df_data.shape[1]

# ==========================================
# ======== Normalize the data ==============
# ==========================================
train_mean = train_df.mean()
# print(train_mean)
train_std = train_df.std()
train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std
print("Train: {}, Val: {}, Test: {}".format(train_df.shape, val_df.shape, test_df.shape))

df_data_std = (df_data - train_mean) / train_std
df_data_std = df_data_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_data_std)
_ = ax.set_xticklabels(df_data.keys(), rotation=90)
plt.savefig('./test/time_series_mulit_step/test_1.png')
plt.close()


OUT_STEPS = 24
multi_window = WindowGenerator(input_width=24, label_width=OUT_STEPS, shift=OUT_STEPS,
                               train_df=train_df, val_df=val_df, test_df=test_df)
print(multi_window)
multi_window.plot('./test/time_series_mulit_step/test_2.png')

# Results
multi_val_performance = {}
multi_performance = {}

# ============================================
# ============= Baseline =====================
# ============================================
class MultiStepLastBaseline(tf.keras.Model):
    def call(self, inputs):
        return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])
last_baseline = MultiStepLastBaseline()
last_baseline.compile(loss=tf.losses.MeanSquaredError(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

multi_val_performance['Baseline'] = last_baseline.evaluate(multi_window.val)
multi_performance['Baseline'] = last_baseline.evaluate(multi_window.test, verbose=0)
multi_window.plot('./test/time_series_mulit_step/test_3.png', last_baseline)

# ============================================
# ========== Repeat Baseline =================
# ============================================
class RepeatBaseline(tf.keras.Model):
  def call(self, inputs):
    return inputs

repeat_baseline = RepeatBaseline()
repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
                        metrics=[tf.metrics.MeanAbsoluteError()])
multi_val_performance['Repeat'] = repeat_baseline.evaluate(multi_window.val)
multi_performance['Repeat'] = repeat_baseline.evaluate(multi_window.test, verbose=0)
multi_window.plot('./test/time_series_mulit_step/test_4.png', repeat_baseline)


MAX_EPOCHS = 20
def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history

# ============================================
# =========== Linear Model ===================
# ============================================
multi_linear_model = tf.keras.Sequential([
    # Take the last time-step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])
# history = compile_and_fit(multi_linear_model, multi_window)
# multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val)
# multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test, verbose=0)
# multi_window.plot('./test/time_series_mulit_step/test_5.png', multi_linear_model)

# ============================================
# =========== Dense Model ===================
# ============================================
multi_dense_model = tf.keras.Sequential([
    # Take the last time step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(512, activation='relu'),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

# history = compile_and_fit(multi_dense_model, multi_window)
# multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val)
# multi_performance['Dense'] = multi_dense_model.evaluate(multi_window.test, verbose=0)
# multi_window.plot('./test/time_series_mulit_step/test_6.png', multi_dense_model)

# ============================================
# =========== CNN Model =====================
# ============================================
CONV_WIDTH = 3
multi_conv_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
    # Shape => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])
#
# history = compile_and_fit(multi_conv_model, multi_window)
# multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
# multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)
# multi_window.plot('./test/time_series_mulit_step/test_7.png', multi_conv_model)

# ============================================
# =========== RNN Model =====================
# ============================================
multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

for i in multi_window.train.take(1):
    print(i)

history = compile_and_fit(multi_lstm_model, multi_window)
multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
multi_window.plot('./test/time_series_mulit_step/test_8.png', multi_lstm_model)


x = np.arange(len(multi_performance))
width = 0.3

metric_name = 'mean_absolute_error'
metric_index = multi_lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in multi_val_performance.values()]
test_mae = [v[metric_index] for v in multi_performance.values()]

plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=multi_performance.keys(),
           rotation=45)
plt.ylabel(f'MAE (average over all times and outputs)')
_ = plt.legend()
plt.savefig('./test/time_series_mulit_step/test_9.png')
plt.close()