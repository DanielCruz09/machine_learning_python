'''This program showcases how to use decision trees in Python.'''

'''------------------Training and Visualizing a Decision Tree--------------------'''

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris(as_frame=True)
x_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(x_iris, y_iris)

from sklearn.tree import export_graphviz

export_graphviz(
    tree_clf,
    out_file="iris_tree.dot",
    feature_names=["petal length (cm)", "petal width (cm)"],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)

# from graphviz import Source

# Source.from_file("iris_tree.dot")

'''----------------------------Making Predictions----------------------------------'''

# print(help(tree_clf.tree_))
# print(tree_clf.predict_proba([[5, 1.5]]).round(3)) # [[0.    0.907 0.093]]
tree_clf_pred = tree_clf.predict([[5, 1.5]])
# print(tree_clf_pred) # [1]

'''------------------------Regularization Hyperparameters--------------------------'''

from sklearn.datasets import make_moons

x_moons, y_moons = make_moons(n_samples=150, noise=0.2, random_state=42)

tree_clf1 = DecisionTreeClassifier(random_state=42)
tree_clf2 = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
tree_clf1.fit(x_moons, y_moons)
tree_clf2.fit(x_moons, y_moons)

x_moons_test, y_moons_test = make_moons(n_samples=1000, noise=0.2, random_state=43)
# print(tree_clf1.score(x_moons_test, y_moons_test)) # 0.898
# print(tree_clf2.score(x_moons_test, y_moons_test)) # 0.92

'''-------------------------Decision Tree Regression---------------------------------'''

import numpy as np
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
x_quad = np.random.rand(200, 1) - 0.5 # a single random input feature
y_quad = x_quad ** 2 + 0.025 * np.random.randn(200, 1)

tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(x_quad, y_quad)

'''-----------------------Sensitivity to Axis Orientation-----------------------------'''

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pca_pipeline = make_pipeline(StandardScaler(), PCA())
x_iris_rotated = pca_pipeline.fit_transform(x_iris)
tree_clf_pca = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf_pca.fit(x_iris_rotated, y_iris)