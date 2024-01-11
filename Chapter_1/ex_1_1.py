# Example 1-1. Training and running a linear model using Scikit-Learn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Download and prepare the data
# UPDATE: This root is no longer active
# data_root = "https://github.com/ageron/data/raw/main"
# lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")

# We will use the data from https://ourworldindata.org/happiness-and-life-satisfaction
lifesat = pd.read_csv("C:\\Users\Daniel Cruz\Machine_Learning_OReilly\gdp-vs-happiness.csv")
X = lifesat[["GDP per capita, PPP (constant 2017 international $)"]].values
Y = lifesat[["Cantril ladder score"]].values

# Visualize the data
lifesat.plot(kind='scatter', grid=True,
             x="GDP per capita, PPP (constant 2017 international $)", y="Cantril ladder score")
plt.axis([23_500, 62_500, 4, 9])
plt.show()

# Select a linear model
model = LinearRegression()

# Train the model
# Since there are NaN values in the data, it is dificult to fit our model to this data; some preprocessing is required
model.fit(X, Y)

# Make a prediction for Cyprus
X_new = [[37_655.2]] # Cyprus's GDP per capita in 2020
print(model.predict(X_new)) # Output: 6.30165767