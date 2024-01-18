'''This program showcases ensemble learning and how to use random forests.'''

'''---------------------------Voting Classifier------------------------------'''

from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

x, y = make_moons(n_samples=500, noise=0.30, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(random_state=42))
    ]
)
voting_clf.fit(x_train, y_train)

# for name, clf in voting_clf.named_estimators_.items():
#     print(name, "=", clf.score(x_test, y_test)) # lr = 0.864, rf = 0.896, svc = 0.896

voting_clf_pred = voting_clf.predict(x_test[:1])
# print(voting_clf_pred) # [1]
# [print(clf.predict(x_test[:1])) for clf in voting_clf.estimators_] # [1], [1], [0]
# print(voting_clf.score(x_test, y_test)) # 0.912

voting_clf.voting = "soft"
voting_clf.named_estimators["svc"].probability = True
voting_clf.fit(x_train, y_train)
# print(voting_clf.score(x_test, y_test)) # 0.92

'''--------------------------Bagging and Pasting------------------------------------'''

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                            max_samples=100, n_jobs=-1) # The -1 tells sklearn to use all available CPU cores
bag_clf.fit(x_train, y_train)

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                            oob_score=True, n_jobs=-1, random_state=42)
bag_clf.fit(x_train, y_train)
# print(bag_clf.oob_score_) # 0/896

from sklearn.metrics import accuracy_score

y_pred = bag_clf.predict(x_test)
bag_clf_acc = accuracy_score(y_test, y_pred)
# print(bag_clf_acc) # 0.92
# print(bag_clf.oob_decision_function_[:3]) # probas for the first 3 instances

'''----------------------------------Random Forests-----------------------------------'''

from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16,
                                 n_jobs=-1, random_state=42)
rnd_clf.fit(x_train, y_train)

y_pred_rf = rnd_clf.predict(x_test)

# These two classifiers are equivalent
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_features="sqrt", max_leaf_nodes=16),
    n_estimators=500, n_jobs=-1, random_state=42
)

from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
rnd_clf.fit(iris.data, iris.target)
# for score, name in zip(rnd_clf.feature_importances_, iris.data.columns):
#     print(round(score, 2), name)

'''---------------------------------Boosting-------------------------------------------'''

from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=30,
    learning_rate=0.5, random_state=42
)
ada_clf.fit(x_train, y_train)

import numpy as np
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
x = np.random.rand(100, 1) - 0.5
y = 3 * x[:, 0] ** 2 + 0.05 * np.random.randn(100) # y = 3x^2 + Gaussian noise

tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(x, y)

y2 = y - tree_reg1.predict(x)
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=43)
tree_reg2.fit(x, y2)

y3 = y2 - tree_reg2.predict(x)
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=44)
tree_reg3.fit(x, y3)

x_new = np.array([[-0.4], [0.], [0.5]])
# print(sum(tree.predict(x_new) for tree in (tree_reg1, tree_reg2, tree_reg3))) # [0.49484029 0.04021166 0.75026781]

from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3,
                                 learning_rate=1.0, random_state=42)
gbrt.fit(x, y)

# Early stopping if the last 10 trees did not help
gbrt_best = GradientBoostingRegressor(
    max_depth=2, learning_rate=0.05, n_estimators=500,
    n_iter_no_change=10, random_state=42
)
gbrt_best.fit(x, y)
# print(gbrt_best.n_estimators_) # 92

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

hgb_reg = make_pipeline(
    make_column_transformer((OrdinalEncoder(), ["ocean_proximity"]),
                            remainder="passthrough"),
                            HistGradientBoostingRegressor(categorical_features=[0], random_state=42)
)
# hgb_reg.fit(housing, housing_labels) # fit on California housing data

'''------------------------------------Stacking--------------------------------------------'''

from sklearn.ensemble import StackingClassifier

stacking_clf = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ],
    final_estimator=RandomForestClassifier(random_state=43),
    cv=5 # number of cross-validation folds
)
stacking_clf.fit(x_train, y_train)