'''This program will classify images as digits.'''

from sklearn.datasets import fetch_openml

# Load in our data
mnist = fetch_openml('mnist_784', as_frame=False, parser='auto') # as_frame specifies whether we want this data as a pandas df or numpy array

x,y = mnist.data, mnist.target

import matplotlib.pyplot as plt

def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")

some_digit = x[0]
# plot_digit(some_digit)
# plt.show()

# Luckily for us, the dataset is already split and shuffled
x_train, x_test = x[:60000], x[60000:]
y_train, y_test = y[:60000], y[60000:]

'''---------------Training a Binary Classifier-------------'''

# Let's simplify the problem and try to identify one digit
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train_5)
# print(sgd_clf.predict([some_digit])) # prints true

'''-----------------Performance Measures-------------------'''

# Cross-Validation

from sklearn.model_selection import cross_val_score

# print(cross_val_score(sgd_clf, x_train, y_train, cv=3, scoring="accuracy"))

from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier()
dummy_clf.fit(x_train, y_train_5)
# print(any(dummy_clf.predict(x_train))) # prints false
# print(cross_val_score(dummy_clf, x_train, y_train_5, cv=3, scoring="accuracy")) # [0.90965 0.90965 0.90965]

# Implementing cross-validation on our own

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

def cross_val():
    skfolds = StratifiedKFold(n_splits=3) # set shuffle=True if data is not already sorted

    for train_index, test_index in skfolds.split(x_train, y_train_5):
        clone_clf = clone(sgd_clf)
        x_train_folds = x_train[train_index]
        y_train_folds = y_train_5[train_index]
        x_test_fold = x_train[test_index]
        y_test_fold = y_train_5[test_index]

        clone_clf.fit(x_train_folds, y_train_folds)
        y_pred = clone_clf.predict(x_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print(n_correct / len(y_pred)) # prints 0.95035, 0.96035, and 0.9604

'''-------------------Confusion Matrices-----------------------'''

from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_train_5, y_train_pred)
# print(cm) # prints [[53892, 687] 
#                     [1891, 3530]]

'''--------------------Precision and Recall----------------------'''

from sklearn.metrics import precision_score, recall_score, f1_score

# print("The precision score is :", precision_score(y_train_5, y_train_pred)) # 0.8370879772350012
# print("The recall score is: " ,recall_score(y_train_5, y_train_pred)) # 0.6511713705958311
# print("The F1 score is: ", f1_score(y_train_5, y_train_pred)) #  0.7325171197343846

# Testing the Precision/Recall Trade-Off

y_scores = sgd_clf.decision_function([some_digit])
# print(y_scores)
threshold = 50
y_some_digit_pred = (y_scores > threshold)
# print(y_some_digit_pred)
y_scores = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3, method="decision_function")

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_functions():
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold")
    [...]  # beautify the figure: add grid, legend, axis, labels, and circles
    plt.show()

def plot_precision_against_recall():
    plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
    plt.show()

# Let's find the threshold that gives us at least 90% precision
idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]
# print(f'{threshold_for_90_precision} gives us at least 90% precision') # 3370.0194991439594

y_train_pred_90 = (y_scores >= threshold_for_90_precision)
# print(precision_score(y_train_5, y_train_pred_90)) # 0.9000345901072293
recall_at_90_precision = recall_score(y_train_5, y_train_pred_90)
# print(recall_at_90_precision) # 0.4799852425751706

'''------------------------The ROC Curve-----------------------------'''

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve():
    idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
    tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

    plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
    plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
    plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")
    plt.show()

from sklearn.metrics import roc_auc_score

area_under_curve = roc_auc_score(y_train_5, y_scores)
# print("The ROC AUC is: ", area_under_curve) #  0.9604938554008616

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)

y_probas_forest = cross_val_predict(forest_clf, x_train, y_train_5, cv=3, method="predict_proba")
# print(y_probas_forest[:2])  # [[0.11 0.89]
                               # [0.99 0.01]]

y_scores_forest = y_probas_forest[:, 1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(
    y_train_5, y_scores_forest)

def plot_pr_forest():
    plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2,
         label="Random Forest")
    plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")
    plt.show()

y_train_pred_forest = y_probas_forest[:, 1] >= 0.5
# print(f1_score(y_train_5, y_train_pred_forest)) # 0.9274509803921569
# print(roc_auc_score(y_train_5, y_scores_forest)) # 0.9983436731328145

'''------------------------Multiclass Classification----------------------'''

from sklearn.svm import SVC

svm_clf = SVC(random_state=42)
svm_clf.fit(x_train[:2000], y_train[:2000]) # train the first 2000 images
svm_predictions = svm_clf.predict([some_digit])
# print(svm_predictions)
some_digit_scores = svm_clf.decision_function([some_digit])
# print(some_digit_scores.round(2)) # [[ 3.79  0.73  6.06  8.3  -0.29  9.3   1.75  2.77  7.21  4.82]]
class_id = some_digit_scores.argmax()
# print(class_id) # 5
# print(svm_clf.classes_[class_id]) # 5

from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(x_train[:2000], y_train[:2000])
ovr_predictions = ovr_clf.predict([some_digit])
# print(ovr_predictions, len(ovr_clf.estimators_)) # ['5'] 10

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train)
sgd_clf.predict([some_digit])
# print(sgd_clf.decision_function([some_digit]).round(2))
# print(cross_val_score(sgd_clf, x_train, y_train, cv=3, scoring="accuracy"))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype("float64"))
# print(cross_val_score(sgd_clf, x_train_scaled, y_train, cv=3, scoring="accuracy"))

'''-----------------------Error Analysis-------------------------------'''

from sklearn.metrics import ConfusionMatrixDisplay

# The images should be on the main diagonal, meaning they were classified correctly
y_train_pred = cross_val_predict(sgd_clf, x_train_scaled, y_train, cv=3)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
# plt.show()

# Making the errors stand out more
sample_weight = (y_train_pred != y_train) 
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        sample_weight=sample_weight, 
                                        normalize="true", values_format=".0%")
# plt.show()

'''---------------------Multilabel Classification-------------------------'''

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Checking if a digit is large (greater or equal to 7)
y_train_large = (y_train >= '7')
# Checking if a digit is odd
y_train_odd = (y_train.astype('int8') % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train, y_multilabel)
knn_clf.predict([some_digit])
y_train_knn_pred = cross_val_predict(knn_clf, x_train, y_multilabel, cv=3)
# print(f1_score(y_multilabel, y_train_knn_pred, average="macro")) # 0.976410265560605

from sklearn.multioutput import ClassifierChain

chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
chain_clf.fit(x_train[:2000], y_train[:2000])
chain_clf.predict([some_digit])

'''------------------------Multioutput Classification---------------------------'''

# Creating a clean image from a noisy one
np.random.seed(42)
noise = np.random.randint(0, 100, (len(x_train), 784))
x_train_mod = x_train + noise
noise = np.random.randint(0, 100, (len(x_test), 784))
x_test_mod = x_test + noise
y_train_mod = x_train
y_test_mod = x_test

knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train_mod, y_train_mod)
clean_digit = knn_clf.predict([x_test_mod[0]])
plot_digit(clean_digit)
# plt.show()

