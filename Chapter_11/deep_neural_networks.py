'''This program showcases how to train deep neural networks.'''

'''--------------------------Vanishing/Exploding Gradient Descent-----------------------------'''

#---------------------------Glorot and He Initialization--------------------------

import tensorflow as tf

dense = tf.keras.layers.Dense(50, activation="relu", 
                              kernel_initializer="he_normal")
he_avg_init = tf.keras.initializers.VarianceScaling(scale=2., mode="fan_avg",
                                                    distribution="uniform")
dense = tf.keras.layers.Dense(50, activation="sigmoid",
                              kernel_initializer=he_avg_init)

#--------------------------------LeakyReLU and PReLU--------------------------------

leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2) # defaults to alpha=0.3
dense = tf.keras.layers.Dense(50, activation=leaky_relu,
                              kernel_initializer="he_normal")
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(50, kernel_initializer="he_normal"),  # no activation
    tf.keras.layers.LeakyReLU(alpha=0.2),  # activation as a separate layer
])

#------------------------------------ELU and SELU----------------------------------------

dense = tf.keras.layers.Dense(50, activation="elu",
                               kernel_initializer="he_normal")
dense = tf.keras.layers.Dense(50, activation="selu",
                              kernel_initializer="lecun_normal") # LeCun initialization is required for SELU

#-----------------------------------GELU, Swish, and Mish--------------------------------

dense = tf.keras.layers.Dense(50, activation="gelu",
                              kernel_initializer="he_normal")
dense = tf.keras.layers.Dense(50, activation="swish",
                               kernel_initializer="he_normal")

#------------------------------------Batch Normalization------------------------------------

tf.keras.backend.clear_session() # reset layer names
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(300, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(100, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation="softmax") # output layer
])
# print(model.summary())
# print([(var.name, var.trainable) for var in model.layers[1].variables])

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

#----------------------------------------Gradient Clipping----------------------------------

optimizer = tf.keras.optimizers.SGD(clipvalue=0.1)
# model.compile([...], optimizer=optimzer)

'''-------------------------------Transfer Learning with Keras----------------------------------------------'''

def transfer_learn(model_name:str, train_data:tuple, valid_data:tuple, test_data:tuple):
    x_train_B, y_train_B = train_data
    x_valid_B, y_valid_B = valid_data
    x_test_B, y_test_B = test_data
    model_A = tf.keras.models.load_model(model_name)
    model_B_on_A = tf.keras.Sequential(model_A.layers[:-1]) # using all layers but the output
    model_B_on_A.add(tf.keras.layers.Dense(1, activation="sigmoid")) # adding the output layer

    model_A_clone = tf.keras.models.clone_model(model_A)
    model_A_clone.set_weights(model_A.get_weights())

    # freeze some layers
    for layer in model_B_on_A.layers[:-1]:
        layer.trainable = False

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer,
                        metrics=["accuracy"])

    history = model_B_on_A.fit(x_train_B, y_train_B, epochs=4,
                            validation_data=(x_valid_B, y_valid_B))

    for layer in model_B_on_A.layers[:-1]:
        layer.trainable = True

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.0005)
    model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer,
                        metrics=["accuracy"])
    history = model_B_on_A.fit(x_train_B, y_train_B, epochs=4,
                            validation_data=(x_valid_B, y_valid_B))
    model_B_on_A.evaluate(x_test_B, y_test_B)

'''------------------------------------------Momentum--------------------------------------------------------'''

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

'''---------------------------------------Learning Schedule--------------------------------------------------'''

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-4) # power scheduling

def exponential_decay_fn(epoch):
    return 0.01 * 0.1 ** (epoch / 20)

def exponential_decay(lr0, s): # lr0 is the initial learning rate
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 ** (epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(lr0=0.01, s=20)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
# history = model.fit(x_train, y_train, [...], callbacks=[lr_scheduler])
# print(history.history["lr"])

def exponential_decay_fn(epoch, lr):
    return lr * 0.1 ** (1 / 20)

def piecewise_constant_fn(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    return 0.001

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(fsctor=0.5, patience=5)
# history = model.fit(x_train, y_train, [...], callbacks=[lr_scheduler])

import math

def compute_lr(x_train):
    batch_size = 32
    n_epochs = 25
    n_steps = n_epochs * math.ceil(len(x_train) / batch_size)
    scheduled_lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01, decay_steps=n_steps, decay_rate=0.1
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=scheduled_lr)

'''----------------------------------L1 and L2 Regularization----------------------------------------------'''

layer = tf.keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal",
                              kernel_regularizer=tf.keras.regularizers.l2(0.01))

from functools import partial

RegularizedDense = partial(tf.keras.layers.Dense,
                           activation="relu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    RegularizedDense(100),
    RegularizedDense(100),
    RegularizedDense(10, activation="softmax")
])

'''-------------------------------------------------Dropout-------------------------------------------------'''

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(100, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(100, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(rate=0.2),     
    tf.keras.layers.Dense(10, activation="softmax")                
])
# [...] compile and train the model

'''----------------------------------------Monte Carlo Dropout-----------------------------------------------'''

# import numpy as np

# y_probas = np.stack([model(x_test, training=True) for sample in range(100)])
# y_proba = y_probas.mean(axis=0)

class MCDroput(tf.keras.layers.Dropout):
    def call(self, inputs, training=False):
        return super().call(inputs, training=True)
    
'''-----------------------------------------Max-Norm Regularization-------------------------------------------'''

dense = tf.keras.layers.Dense(
    100, activation="relu", kernel_initializer="he_normal",
    kernel_constraint=tf.keras.constrains.max_norm(1.)
)
