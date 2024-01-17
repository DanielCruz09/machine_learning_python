'''This program showcases SVMs.'''

'''-------------------------Linear SVM Classifiers------------------------------'''

from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 2) # Iris virginica

svm_clf = make_pipeline(StandardScaler(),
                        LinearSVC(C=1, random_state=42, dual='auto'))
svm_clf.fit(X, y)

X_new = [[5.5, 1.7], [5.0, 1.5]]
svm_pred = svm_clf.predict(X_new)
# print(svm_pred) # [ True False]
# print(svm_clf.decision_function(X_new)) # [ 0.66163816 -0.22035761]

'''------------------------Nonlinear SVM Classification--------------------------'''

from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures

X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

polynomial_svm_clf = make_pipeline(
    PolynomialFeatures(degree=3),
    StandardScaler(),
    LinearSVC(C=10, max_iter=10_000, random_state=42, dual='auto')
)
polynomial_svm_clf.fit(X, y)

'''-----------------------------Polynomial kernel-----------------------------------'''

from sklearn.svm import SVC

poly_kernel_svm_clf = make_pipeline(StandardScaler(),
                                    SVC(kernel="poly", degree=3, coef0=1, C=5))
poly_kernel_svm_clf.fit(X, y)

'''----------------------------Gaussian RBF Kernel-----------------------------------'''

rbf_kernel_svm_clf = make_pipeline(StandardScaler(),
                                   SVC(kernel="rbf", gamma=5, C=0.001))
rbf_kernel_svm_clf.fit(X, y)

'''----------------------------------SVM Regression------------------------------------'''

from sklearn.svm import LinearSVR, SVR

svm_reg = make_pipeline(StandardScaler(),
                        LinearSVR(epsilon=0.5, random_state=42, dual='auto'))
svm_reg.fit(X, y)

# For nonlinear datasets

svm_poly_reg = make_pipeline(StandardScaler(),
                             SVR(kernel="poly", degree=2, C=0.01, epsilon=0.1))
svm_poly_reg.fit(X, y)
