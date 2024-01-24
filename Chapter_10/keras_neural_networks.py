'''This program offers an introduction to artificial neural networks in keras'''

'''-----------------------------Perceptron-------------------------------------'''

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris(as_frame=True)
x = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 0) # Iris setosa

per_clf = Perceptron(random_state=42)
per_clf.fit(x, y)

x_new = [[2, 0.5], [3, 1]]
y_pred = per_clf.predict(x_new) # predicts True and False for these 2 flowers

'''------------------------------Regrssion MLPs-------------------------------'''

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
x_train_full, x_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42
)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_full, y_train_full, random_state=42
)

mlp_reg = MLPRegressor(hidden_layer_sizes=[50, 50, 50], random_state=42)
pipeline = make_pipeline(StandardScaler(), mlp_reg)
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_valid)
rmse = mean_squared_error(y_valid, y_pred, squared=False) # about 5.05

'''----------------------Image Classifier using Keras-----------------------------'''

import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist
x_train, y_train = x_train_full[:-5000], y_train_full[:-5000]
x_valid, y_valid = x_train_full[-5000:], y_train_full[-5000:]

# print(x_train.shape) # 28x28 array
# print(x_train.dtype) # uint8

x_train, x_valid, x_test, = x_train / 255., x_valid / 255., x_test / 255.
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# print(class_names[y_train[0]]) # Ankle boot

'''--------------------Creating the Model using Sequantial API--------------------'''

# Adding two hidden layers
tf.random.set_seed(42)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=[28, 28]))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(300, activation="relu"))
model.add(tf.keras.layers.Dense(100, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# print(model.summary())
# print(model.layers)
hidden1 = model.layers[1]
# print(hidden1.name)
# print(model.get_layer('dense') is hidden1)

weights, biases = hidden1.get_weights()
# print(weights, biases)
# print(weights.shape, biases.shape)

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd", metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=30, verbose=0,
                    validation_data=(x_valid, y_valid))

import matplotlib.pyplot as plt
import pandas as pd

pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[0, 29], ylim=[0, 1], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"]
)
# plt.show()

model.evaluate(x_test, y_test, verbose=0)

'''------------------Using the Model to Make Predictions---------------------'''

x_new = x_test[:3]
y_proba = model.predict(x_new)
# print(y_proba.round(2))

y_pred = y_proba.argmax(axis=-1)
# print(y_pred) # [9,2,1]
# print(np.array(class_names)[y_pred]) # ['Ankle boot', 'Pullover', 'Trouser']

'''--------------Building a Regression MLP using Sequential API----------------'''

tf.random.set_seed(42)
norm_layer = tf.keras.layers.Normalization(input_shape=x_train.shape[1:])
model = tf.keras.Sequential([
    norm_layer,
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(1)
])
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
norm_layer.adapt(x_train)
history = model.fit(x_train, y_train, epochs=20, verbose=0,
                    validation_data=(x_valid, y_valid))
mse_test, rmse_test = model.evaluate(x_test, y_test)
x_new = x_test[:3]
y_pred = model.predict(x_new)

'''----------------Building Complex Models Using the Functional API---------------'''

normalization_layer = tf.keras.layers.Normalization()
hidden_layer1 = tf.keras.layers.Dense(30, activation="relu")
hidden_layer2 = tf.keras.layers.Dense(30, activation="relu")
concat_layer = tf.keras.layers.Concatenate()
output_layer = tf.keras.layers.Dense(1)

input_ = tf.keras.layers.Input(shape=x_train.shape[1:])
normalized = normalization_layer(input_)
hidden1 = hidden_layer1(normalized)
hidden2 = hidden_layer2(hidden1)
concat = concat_layer([normalized, hidden2])
output = output_layer(concat)

model = tf.keras.Model(inputs=[input_], outputs=[output])

# Handling multiple inputs
input_wide = tf.keras.layers.Input(shape=[5]) # features 0 to 4
input_deep = tf.keras.layers.Input(shape=[6]) # features 2 to 7
norm_layer_wide = tf.keras.layers.Normalization()
norm_layer_deep = tf.keras.layers.Normalization()
norm_wide = norm_layer_wide(input_wide)
norm_deep = norm_layer_deep(input_deep)
hidden1 = tf.keras.layers.Dense(30, activation="relu")(norm_deep)
hidden2 = tf.keras.layers.Dense(30, activation="relu")(hidden1)
concat = tf.keras.layers.concatenate([norm_wide, hidden2])
output = tf.keras.layers.Dense(1)(concat)
model = tf.keras.Model(inputs=[input_wide, input_deep], outputs=[output])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])

x_train_wide, x_train_deep = x_train[:, :5], x_train[:, 2:]
x_valid_wide, x_valid_deep = x_valid[:, :5], x_valid[:, 2:]
x_test_wide, x_test_deep = x_test[:, :5], x_test[:, 2:]
x_new_wide, x_new_deep = x_test_wide[:3], x_test_deep[:3]

def make_func_model():
    norm_layer_wide.adapt(x_train_wide)
    norm_layer_deep.adapt(x_train_deep)
    history = model.fit((x_train_wide, x_train_deep), y_train, epochs=20,
                        verbose=0, validation_data=((x_valid_wide, x_valid_deep), y_valid))
    mse_test = model.evaluate((x_test_wide, x_test_deep), y_test, verbose=0)
    y_pred = model.predict((x_new_wide, x_new_deep))

    # Adding more output
    output = tf.keras.layers.Dense(1)(concat)
    aux_output = tf.keras.layers.Dense(1)(hidden2)
    model = tf.keras.Model(inputs=[input_wide, input_deep],
                        outputs=[output, aux_output])
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss=("mse", "mse"), loss_weights=(0.9, 0.1), optimizer=optimizer,
                metrics=["RootMeanSquaredError"])
    norm_layer_wide.adapt(x_train_wide)
    norm_layer_deep.adapt(x_train_deep)
    history = model.fit(
        (x_train_wide, x_train_deep), (y_train, y_train), epochs=20, verbose=0,
        validation_data=((x_valid_wide, x_valid_deep), (y_valid, y_valid))
    )
    eval_results = model.evaluate((x_test_wide, x_test_deep), (y_test, y_test), verbose=0)
    weighted_sum_of_losses, main_loss, aux_loss, main_rmse, aux_rmse = eval_results
    y_pred_main, y_pred_aux = model.predict((x_new_wide, x_new_deep))

    y_pred_tuple = model.predict((x_new_wide, x_new_deep))
    y_pred = dict(zip(model.output_names, y_pred_tuple))

'''----------------------------Using the Subclassing API for Dynamic Models-----------------------'''

class WideAndDeepModel(tf.keras.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs) # needed to support naming the model
        self.norm_layer_wide = tf.keras.layers.Normalization()
        self.norm_layer_deep = tf.keras.layers.Normalization()
        self.hidden1 = tf.keras.layers.Dense(units, activation=activation)
        self.hidden2 = tf.keras.layers.Dense(units, activation=activation)
        self.main_output = tf.keras.layers.Dense(1)
        self.aux_output = tf.keras.layers.Dense(1)

    def call(self, inputs:tuple):
        input_wide, input_deep = inputs
        norm_wide = self.norm_layer_wide(input_wide)
        norm_deep = self.norm_layer_deep(input_deep)
        hidden1 = self.hidden1(norm_deep)
        hidden2 = self.hidden2(hidden1)
        concat = tf.keras.layers.concatenate([norm_wide, hidden2])
        output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return output, aux_output
    
model = WideAndDeepModel(30, activation="relu", name="my_cool_model")

'''------------------------------------Saving and Restoring a Model------------------------------------'''

model.save("my_keras_model", save_format="tf") # to save
model = tf.keras.models.load_model("my_keras_model") # to load
y_pred_main, y_pred_aux = model.predict((x_new_wide, x_new_deep)) # use as normally

'''---------------------------------------Using Callbacks---------------------------------------------'''

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("my_checkpoints", save_weights_only=True)
# history = model.fit([...], callbacks=[checkpoint_cb])

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
# history = model.fit([...], callbacks=[checkpoint_cb, early_stopping_cb])

class PrintValTrainRatioCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        ratio = logs["val_loss"] / logs["loss"]
        print(f'Epoch={epoch}, val/train={ratio:.2f}')

'''------------------------------------Visualization with TensorBoard-------------------------------------'''

from pathlib import Path
from time import strftime

def get_run_logdir(root_logdir="my_logs"):
    return Path(root_logdir) / strftime("run_%Y_%m_%d_%H_%M_%S")

run_logdir = get_run_logdir() # e.g., my_logs/run_today's date

tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir, profile_batch=(100, 200))
# history = model.fit([...], callbacks=[tensorboard_cb])

test_logdir = get_run_logdir()
writer = tf.summary.create_file_writer(str(test_logdir))
with writer.as_default():
    for step in range(1, 1000 + 1):
        tf.summary.scalar("my_scalar", np.sin(step / 10), step=step)

        data = (np.random.randn(100) + 2) * step / 100  # gets larger
        tf.summary.histogram("my_hist", data, buckets=50, step=step)

        images = np.random.rand(2, 32, 32, 3) * step / 1000  # gets brighter
        tf.summary.image("my_images", images, step=step)

        texts = ["The step is " + str(step), "Its square is " + str(step ** 2)]
        tf.summary.text("my_text", texts, step=step)

        sine_wave = tf.math.sin(tf.range(12000) / 48000 * 2 * np.pi * step)
        audio = tf.reshape(tf.cast(sine_wave, tf.float32), [1, -1, 1])
        tf.summary.audio("my_audio", audio, sample_rate=48000, step=step)

'''--------------------------Fine-Tuning Neural Network Hyperparameters--------------------------'''

import keras_tuner as kt

def build_model(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=8, default=2)
    n_neurons = hp.Int("n_neurons", min_value=16, max_value=256)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2,
                             sampling="log")
    optimizer = hp.Choice("optimizer", values=["sgd", "adam"])
    if optimizer == "sgd":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model = tf.keras.Sequantial()
    model.add(tf.keras.layers.Flatten())
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax")) # output layer
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])
    return model

random_search_tuner = kt.RandomSearch(
    build_model, objective="val_accuracy", max_trials=5, overwrite=True,
    directory="my_fashion_mnist", project_name="my_rnd_search", seed=42
)
random_search_tuner.search(x_train, y_train, epochs=10, validation_data=(x_valid, y_valid))

top3_models = random_search_tuner.get_best_models(num_models=3)
best_model = top3_models[0]
top3_params = random_search_tuner.get_best_hyperparameters(num_trials=3)
print(top3_params[0].values)
best_trial = random_search_tuner.oracle.get_best_trials(num_trials=1)[0]
print(best_trial.summary())
print(best_trial.metrics.get_last_value("vall_accuracy"))

best_model.fit(x_train_full, y_train_full, epochs=10)
test_loss, test_accuracy = best_model.evaluate(x_test, y_test)

class MyClassificationHyperModel(kt.HyperModel):
    def build(self, hp):
        return build_model(hp)
    
    def fit(self, hp, model, x, y, **kwargs):
        if hp.Boolean("normalize"):
            norm_layer = tf.keras.layers.Normalization()
            x = norm_layer(x)
        return model.fit(x, y, **kwargs)
    
hyperband_tuner = kt.Hyperband(
    MyClassificationHyperModel(), objective="val_accuracy", seed=42,
    max_epochs=10, factor=3, hyperband_iterations=2,
    overwrite=True, directory="my_fashion_mnist", project_name="hyperband"
)

root_logdir = Path(hyperband_tuner.project_dir) / "tensorboard"
tensorboard_cb = tf.keras.callbacks.TensorBoard(root_logdir)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=2)
hyperband_tuner.search(x_train, y_train, epochs=10,
                       validation_data=(x_valid, y_valid),
                       callbacks=[early_stopping_cb, tensorboard_cb])

bayesian_opt_tuner = kt.BayesianOptimization(
    MyClassificationHyperModel(), objective="val_accuracy", seed=42,
    max_trials=10, alpha=1e-4, beta=2.6,
    overwrite=True, directory="my_fashion_mnist", project_name="bayesian_opt"
)
# bayesian_opt_tuner.search([...])