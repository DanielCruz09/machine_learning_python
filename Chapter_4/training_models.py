'''This program will look in-depth into how ML training algorithms function.'''

'''------------------------The Normal Equation-------------------------------'''

import numpy as np

np.random.seed(42)
m = 100 # number of instances
X = 2 * np.random.rand(m, 1) # column vector
y = 4 + 3 * X + np.random.rand(m, 1) # column vector

from sklearn.preprocessing import add_dummy_feature

# compute the inverse of a matrix
X_b = add_dummy_feature(X) # add X0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y # @ operator performs matrix multiplication
# print(theta_best) # [[4.51359766]
                     # [2.98323418]]

# Make new predictions using theta
X_new = np.array([[0], [2]])
X_new_b = add_dummy_feature(X_new) # add x0 = 1 to each instance
y_predict = X_new_b @ theta_best
# print(y_predict)  # [[ 4.51359766]
                     # [10.48006601]]

import matplotlib.pyplot as plt

def plot_predictions():
    plt.plot(X_new, y_predict, "r-", label="Predictions")
    plt.plot(X, y, "b.")
    plt.show()

# Now, let's do linear regression

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
# print(lin_reg.intercept_, lin_reg.coef_)# [4.51359766] [[2.98323418]] 
# print(lin_reg.predict(X_new)) # [[ 4.51359766]
                                 # [10.48006601]]

theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
# print(theta_best_svd)
# print(np.linalg.pinv(X_b) @ 7)

'''-------------------------------Batch Gradient Descent-------------------------'''

eta = 0.1 # learning rate
n_epochs = 1000
m = len(X_b) # number of instances

np.random.seed(42)
theta = np.random.randn(2, 1) # randonly-initialized model params

for epoch in range(n_epochs):
    gradients = 2 / m * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients

# print(theta) # [[4.51359766]
                # [2.98323418]]
    
'''-------------------------Stochastic Gradient Descent--------------------------------'''

n_epochs = 50
t0, t1 = 5, 50 # learning schedule hypermarameters

def learning_schedule(t):
    return t0 / (t + t1)

np.random.seed(42)
theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    for iteration in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index : random_index + 1]
        yi = y[random_index : random_index + 1]
        gradients = 2 * xi.T @ (xi @ theta - yi) # for SGD, do not divide by m
        eta = learning_schedule(epoch * m + iteration)
        theta = theta - eta * gradients

# print(theta) # [[4.51548062]
                # [2.9775157 ]]
        
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01,
                       n_iter_no_change=100, random_state=42)
sgd_reg.fit(X, y.ravel()) # because fit() expexts 1D targets

# print(sgd_reg.intercept_, sgd_reg.coef_) # [4.50316965] [2.99156535]

'''----------------------------Polynomial Regression--------------------------------'''

np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
# print(X[0]) # [-0.75275929]
# print(X_poly[0]) # [-0.75275929  0.56664654]

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
# print(lin_reg.intercept_, lin_reg.coef_) # [1.78134581] [[0.93366893 0.56456263]]

'''----------------------------Learning Curves-------------------------------------'''

from sklearn.model_selection import learning_curve

train_sizes, train_scores, valid_scores = learning_curve(
    LinearRegression(), X, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
    scoring="neg_root_mean_squared_error"
)
train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)

def plot_learning_curve():
    plt.plot(train_sizes, train_errors, "r-+", linewidth=2, label="train")
    plt.plot(train_sizes, valid_errors, "b-", linewidth=3, label="valid")
    plt.show()

from sklearn.pipeline import make_pipeline

polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=10, include_bias=False),
    LinearRegression()
)

train_sizes, train_scores, valid_scores = learning_curve(
    polynomial_regression, X, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
    scoring="neg_root_mean_squared_error")
train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)

'''---------------------------Ridge Regression------------------------------------'''

from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=0.1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg_pred = ridge_reg.predict([[1.5]])
# print(ridge_reg_pred) # [[4.82899748]]

sgd_reg = SGDRegressor(penalty="l2", alpha=0.1 / m, tol=None,
                       max_iter=1000, eta0=0.01, random_state=42)
sgd_reg.fit(X, y.ravel())
# print(sgd_reg.predict([[1.5]])) # [4.82830117]

'''-------------------------------Lasso Regression----------------------------------'''

from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg_pred = lasso_reg.predict([[1.5]])
# print(lasso_reg_pred)  # [4.77621741]

'''-----------------------------Elastic Net Regression-----------------------------------'''

from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5) # l1_ratio is equivalent to mis ratio r
elastic_net.fit(X, y)
elastic_net_pred = elastic_net.predict([[1.5]])
# print(elastic_net_pred) # [4.78114505]

'''-----------------------------------Early Stopping----------------------------------------'''

from copy import deepcopy
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def early_stop():

    X_train, y_train, X_valid, y_valid = [...]  # split the quadratic dataset

    preprocessing = make_pipeline(PolynomialFeatures(degree=90, include_bias=False),
                                StandardScaler())
    X_train_prep = preprocessing.fit_transform(X_train)
    X_valid_prep = preprocessing.transform(X_valid)
    sgd_reg = SGDRegressor(penalty=None, eta0=0.002, random_state=42)
    n_epochs = 500
    best_valid_rmse = float('inf')

    for epoch in range(n_epochs):
        sgd_reg.partial_fit(X_train_prep, y_train)
        y_valid_predict = sgd_reg.predict(X_valid_prep)
        val_error = mean_squared_error(y_valid, y_valid_predict, squared=False)
        if val_error < best_valid_rmse:
            best_valid_rmse = val_error
            best_model = deepcopy(sgd_reg)

'''-------------------------------Logistic Regression---------------------------------'''

from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = iris.data[["petal width (cm)"]].values
y = iris.target_names[iris.target] == 'virginica'
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

def plot_petal_width_probs():
    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)  # reshape to get a column vector
    y_proba = log_reg.predict_proba(X_new)
    decision_boundary = X_new[y_proba[:, 1] >= 0.5][0, 0]

    plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2,
            label="Not Iris virginica proba")
    plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica proba")
    plt.plot([decision_boundary, decision_boundary], [0, 1], "k:", linewidth=2,
            label="Decision boundary")
    [...] # beautify the figure: add grid, labels, axis, legend, arrows, and samples
    plt.show()

# print(log_reg.predict([[1.7], [1.5]])) # [ True False]
    
'''------------------------------Softmax Regression----------------------------------'''

X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

softmax_reg = LogisticRegression(C=30, random_state=42)
softmax_reg.fit(X_train, y_train)

# print(softmax_reg.predict([[5, 2]])) # [2]
# print(softmax_reg.predict_proba([[5, 2]]).round(2)) # [[0.   0.04 0.96]]
