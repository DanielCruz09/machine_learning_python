'''Processing sequences using RNNs and CNNs.'''

'''---------------------------------------------------Forecasting a Time Series-------------------------------------------------------'''

import pandas as pd
from pathlib import Path

path = Path("CTA_-_Ridership_-_Daily_Boarding_Totals.csv")
df = pd.read_csv(path, parse_dates=["service_date"])
df.columns = ["date", "day_type", "bus", "rail", "total"] # shorter names
df = df.sort_values("date").set_index("date")
df.index = pd.to_datetime(df.index)
df = df.drop("total", axis=1) # no need for total, it's just bus + rail
df = df.drop_duplicates() # remove duplicated months

# print(df.head())

import matplotlib.pyplot as plt

df["2019-03":"2019-05"].plot(grid=True, marker=".", figsize=(8, 3.5))
# plt.show()

diff_7 = df[["bus", "rail"]].diff(7)["2019-03":"2019-05"]
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
df.shift(7).plot(ax=axs[0], grid=True, legend=False, linestyle=":") # lagged
diff_7.plot(ax=axs[1], grid=True, marker=".") # 7-day difference time series
# plt.show()

# print(list(df.loc["2019-05-25":"2019-05-27"]["day_type"])) # ['A', 'U', 'U']

# print(diff_7.abs().mean()) # mean absolute error

targets = df[["bus", "rail"]]["2019-03":"2019-05"]
# print((diff_7 / targets).abs().mean()) # mean absolute percentage error

period = slice("2001", "2019")
df_monthly = df.drop("day_type", axis=1).resample('M').mean() # compute the mean for each month
rolling_avg_12_months = df_monthly[period].rolling(window=12).mean()

fig, ax = plt.subplots(figsize=(8, 4))
df_monthly[period].plot(ax=ax, marker=".")
rolling_avg_12_months.plot(ax=ax, grid=True, legend=False)
# plt.show()

df_monthly.diff(12)[period].plot(grid=True, marker=".", figsize=(8, 3))
# plt.show()

'''---------------------------------------------------------ARMA Model-------------------------------------------------------------------'''

from statsmodels.tsa.arima.model import ARIMA

origin, today = "2019-01-01", "2019-05-31"
rail_series = df.loc[origin:today]["rail"].asfreq("D")
model = ARIMA(rail_series,
              order=(1, 0, 0),
              seasonal_order=(0, 1, 1, 7))
model = model.fit()
y_pred = model.forecast() # returns 427,758.6; in reality, it should be 379,044

origin, start_date, end_date = "2019-01-01", "2019-03-01", "2019-05-31"
time_period = pd.date_range(start_date, end_date)
rail_series = df.loc[origin:end_date]["rail"].asfreq("D")
y_preds = []

def compute_mae():
    for today in time_period.shift(-1):
        model = ARIMA(rail_series[origin:today], #train on data up to "today"
                    order=(1, 0, 0),
                    seasonal_order=(0, 1, 1, 7))
        model = model.fit() # we retrain the model every day
        y_pred = model.forecast()[0]
        y_preds.append(y_pred)

    y_preds = pd.Series(y_preds, index=time_period)
    mae = (y_preds - rail_series[time_period]).abs().mean() # returns 32, 040.7

'''-------------------------------------------------------Preparing the Data---------------------------------------------------------------'''

import tensorflow as tf

my_series = [0, 1, 2, 3, 4, 5]
my_dataset = tf.keras.utils.timeseries_dataset_from_array(
    my_series,
    targets=my_series[3:], # the targets are 3 steps into the future
    sequence_length=3,
    batch_size=2
)
# print(list(my_dataset))

# Alternatively
def print_windows():
    for window_dataset in tf.data.Dataset.range(6).window(4, shift=1):
        for element in window_dataset:
            print(f'{element}', end=" ")
        print()

def print_flattened_windows():
    dataset = tf.data.Dataset.range(6).window(4, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window_dataset: window_dataset.batch(4))
    for window_tensor in dataset:
        print(f'{window_tensor}')

def to_windows(dataset, length):
    dataset = dataset.window(length, shift=1, drop_remainder=True)
    return dataset.flat_map(lambda window_ds: window_ds.batch(length))

dataset = to_windows(tf.data.Dataset.range(6), 4)
dataset = dataset.map(lambda window: (window[:-1], window[-1]))
# print(list(dataset.batch(2)))

rail_train = df["rail"]["2016-01":"2018-12"] / 1e6
rail_valid = df["rail"]["2019-01":"2019-05"] / 1e6
rail_test = df["rail"]["2019-06":] / 1e6

seq_length = 56
train_ds = tf.keras.utils.timeseries_dataset_from_array(
    rail_train.to_numpy(),
    targets=rail_train[seq_length:],
    sequence_length=seq_length,
    batch_size=32,
    shuffle=True,
    seed=42
)
valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    rail_valid.to_numpy(),
    targets=rail_valid[seq_length:],
    sequence_length=seq_length,
    batch_size=32
)

'''------------------------------------------------Forecasting with a Linear Model-------------------------------------------------------'''

tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[seq_length])
])
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_mae", patience=50, restore_best_weights=True
)
opt = tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(), optimizer=opt, metrics=["mae"])
# history = model.fit(train_ds, validation_data=valid_ds, epochs=500,
#                     callbacks=[early_stopping_cb])

'''---------------------------------------------------Forecasting with a Simple RNN-----------------------------------------------------------'''

model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(1, input_shape=[None, 1])
]) # MAE of over 100,000; not good

univar_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape=[None, 1]),
    tf.keras.layers.Dense(1) # no activation function by default
]) # MAE of 27,703; pretty good

'''------------------------------------------------------Forecasting with a Deep RNN----------------------------------------------------------'''

deep_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.SimpleRNN(32, return_sequences=True),
    tf.keras.layers.SimpleRNN(32),
    tf.keras.layers.Dense(1)
]) # MAE of 31,211

'''-----------------------------------------------------Forecasting Multivariate Time Series----------------------------------------------------'''

df_mulvar = df[["bus", "rail"]] / 1e6 # use both bus & rail as input
df_mulvar["next_day_type"] = df["day_type"].shift(-1) # we know tomorrow's type
df_mulvar = pd.get_dummies(df_mulvar) # one-hot encode the day type

mulvar_train = df_mulvar["2016-01":"2018-12"].astype(int)
mulvar_valid = df_mulvar["2019-01":"2019-05"].astype(int)
mulvar_test = df_mulvar["2019-06":].astype(int)

# print(mulvar_train.sample().dtypes)

train_mulvar_ds = tf.keras.utils.timeseries_dataset_from_array(
    mulvar_train.to_numpy(), # use all 5 columns as input
    targets=mulvar_train["rail"][seq_length:], # forecast only until rail series
    sequence_length=seq_length,
    batch_size=32,
    shuffle=True,
    seed=42
)
valid_mulvar_ds = tf.keras.utils.timeseries_dataset_from_array(
    mulvar_valid.to_numpy(),
    targets=mulvar_valid["rail"][seq_length:],
    sequence_length=seq_length,
    batch_size=32
)

mulvar_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape=[None, 5]),
    tf.keras.layers.Dense(1)
]) # MAE of 22,062

# For both the bus and rail
train_mulvar_ds = tf.keras.utils.timeseries_dataset_from_array(
    mulvar_train.to_numpy(), # use all 5 columns as input
    targets=mulvar_train[["bus", "rail"]][seq_length:], # forecast only until rail series
    sequence_length=seq_length,
    batch_size=32,
    shuffle=True,
    seed=42
)
valid_mulvar_ds = tf.keras.utils.timeseries_dataset_from_array(
    mulvar_valid.to_numpy(),
    targets=mulvar_valid[["bus", "rail"]][seq_length:],
    sequence_length=seq_length,
    batch_size=32
)

mulvar_model_bus_rail = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape=[None, 5]),
    tf.keras.layers.Dense(2)
]) # MAE of 25,330 for rail; 26,369 for bus

'''------------------------------------------------Forecasting Several Time Steps Ahead-------------------------------------------------'''

import numpy as np

x = rail_valid.to_numpy()[np.newaxis, :seq_length, np.newaxis] # shape of [1, 56, 1]
for step_ahead in range(14):
    y_pred_one = univar_model.predict(x)
    x = np.concatenate([x, y_pred_one.reshape(1, 1, 1)], axis=1)

def split_inputs_and_targets(mulvar_series, ahead=14, target_col=1):
    return mulvar_series[:, :-ahead], mulvar_series[:, -ahead:, target_col]

ahead_train_ds = tf.keras.utils.timeseries_dataset_from_array(
    mulvar_train.to_numpy(),
    targets=None,
    sequence_length=seq_length + 14,
    batch_size=32,
    shuffle=True,
    seed=42
).map(split_inputs_and_targets)

ahead_valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    mulvar_valid.to_numpy(),
    targets=None,
    sequence_length=seq_length + 14,
    batch_size=32
).map(split_inputs_and_targets)

ahead_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape=[None, 5]),
    tf.keras.layers.Dense(14)
])

x = mulvar_valid.to_numpy()[np.newaxis, :seq_length] # shape [1, 56, 1]
y_pred = ahead_model.predict(x) # shape [1, 14]

'''----------------------------------------------------Forecasting with a Seq2Seq Model-----------------------------------------------------------'''

my_series = tf.data.Dataset.range(7)
dataset = to_windows(to_windows(my_series, 3), 4)
# print(list(dataset))
dataset = dataset.map(lambda S: (S[:, 0], S[:, 1:]))
# print(list(dataset))

def to_seq2seq_dataset(series, seq_length=56, ahead=14, target_col=1,
                       batch_size=32, shuffle=False, seed=None):
    ds = to_windows(tf.data.Dataset.from_tensor_slices(series), ahead + 1)
    ds = to_windows(ds, seq_length).map(lambda S: (S[:, 0], S[:, 1:, 1]))
    if shuffle:
        ds = ds.shuffle(8 * batch_size, seed=seed)
    return ds.batch(batch_size=batch_size)

seq2seq_train = to_seq2seq_dataset(mulvar_train, shuffle=True, seed=42),
seq2seq_valid = to_seq2seq_dataset(mulvar_valid)

seq2seq_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, return_sequences=True, input_shape=[None, 5]),
    tf.keras.layers.Dense(14)
]) # At t + 14, the MAE is 34,322

x = mulvar_valid.to_numpy()[np.newaxis, :seq_length]
y_pred_14 = seq2seq_model.predict(x)[0, -1] # only the last time step's output

'''---------------------------------------------------Fighting the Unstable Gradients Problem------------------------------------------------------'''

class LNSimpleRNNCell(tf.keras.layers.Layer):
    def __init__(self, units, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.state_size = units
        self.output_size = units
        self.simple_rnn_cell = tf.keras.layers.SimpleRNN(units, activation=None)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.activation = tf.keras.activations.get(activation)

    def call(self, inputs, states):
        outputs, new_states = self.simple_rnn_cell(inputs, states)
        norm_outputs = self.activation(self.layer_norm(outputs))
        return norm_outputs, [norm_outputs]
    
custom_ln_model = tf.keras.Sequential([
    tf.keras.layers.RNN(LNSimpleRNNCell(32), return_sequences=True,
                        input_shape=[None, 5]),
    tf.keras.layers.Dense(14)
])

'''---------------------------------------------------------------------LSTM Cells---------------------------------------------------------------'''

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True, input_shape=[None, 5]),
    tf.keras.layers.Dense(14)
])

'''---------------------------------------------Using 1D Convolutional Layers to Process Sequences------------------------------------------------'''

conv_rnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=4, strides=2,
                           activation="relu", input_shape=[None, 5]),
    tf.keras.layers.GRU(32, return_sequences=True),
    tf.keras.layers.Dense(14)
])

longer_train = to_seq2seq_dataset(mulvar_train, seq_length=112,
                                  shuffle=True, seed=42)
longer_valid = to_seq2seq_dataset(mulvar_valid, seq_length=112)
downsampled_train = longer_train.map(lambda x, y: (x, y[:, 3::2]))
downsampled_valid = longer_valid.map(lambda x, y: (x, y[:, 3::2]))
# comple and fit the model using the downsamples datasets

'''-------------------------------------------------------------------------WaveNet---------------------------------------------------------------'''

wavenet_model = tf.keras.Sequential()
wavenet_model.add(tf.keras.layers.Input(shape=[None, 5]))
for rate in (1, 2, 4, 8) * 2:
    wavenet_model.add(tf.keras.layers.Conv1D(
        filters=32, kernel_size=2, padding="causal", activation="relu",
        dilation_rate=rate
    ))
wavenet_model.add(tf.keras.layers.Conv1D(filters=14, kernel_size=1))

