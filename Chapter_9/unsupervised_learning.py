'''This program showcases unsupervised learning techniques.'''

'''---------------------------------k-means-------------------------------------'''

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

x, y = make_blobs()
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
y_pred = kmeans.fit_predict(x)
# print(y_pred) # [4, 0, 3, ..., 0, 4, 1]
# print(y_pred is kmeans.labels_) # True
# print(kmeans.cluster_centers_)

import numpy as np

x_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
kmeans_pred = kmeans.predict(x_new)
# print(kmeans_pred) # [3 2 3 3]
kmeans.transform(x_new).round(2)

#------------------------Centroid initialization methods--------------------------
good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
kmeans = KMeans(n_clusters=5, init=good_init, n_init=1, random_state=42)
kmeans.fit(x)
# print(kmeans.inertia_) # 167.0066626218993
# print(kmeans.score(x)) # -171.7537927433981

#--------------------Accelerated k-means and mini-batch k-means--------------------

from sklearn.cluster import MiniBatchKMeans

minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, n_init='auto')
minibatch_kmeans.fit(x)

# Finding the optimal number of clusters
from sklearn.metrics import silhouette_score

silhouette_score(x, kmeans.labels_) # 0.6772794810777306

#-------------------------------Image Segmentation----------------------------------

import PIL

def image_segmentation(filepath):
    image = np.asarray(PIL.Image.open(filepath))
    # print(image.shape)

    x = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=8, random_state=42, n_init='auto').fit(x)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_img = segmented_img.reshape(image.shape)

#--------------------------------Semi-Supervised Learning------------------------------
    
from sklearn.datasets import load_digits

x_digits, y_digits = load_digits(return_X_y=True)
x_train, y_train = x_digits[:1400], y_digits[:1400]
x_test, y_test = x_digits[1400:], y_digits[1400:]

from sklearn.linear_model import LogisticRegression

n_labeled = 50
log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(x_train[:n_labeled], y_train[:n_labeled])
# print(log_reg.score(x_test, y_test)) # 0.7481108312342569

k = 50
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
x_digits_dist = kmeans.fit_transform(x_train)
representative_digit_idx = np.argmin(x_digits_dist, axis=0)
x_representative_digits = x_train[representative_digit_idx]
digits = [1,3,6,0,7,9,2,4,8,9,
          5,4,7,1,2,6,1,2,5,1,
          4,1,3,3,8,8,2,5,6,9,
          1,4,0,6,8,3,4,6,7,2,
          6,1,0,7,5,1,9,9,3,7]
y_representative_digits = np.array(digits)

log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(x_representative_digits, y_representative_digits)
# print(log_reg.score(x_test, y_test))

y_train_propagated = np.empty(len(x_train), dtype=np.int64)
for i in range(k):
    y_train_propagated[kmeans.labels_ == i] = y_representative_digits[i]

log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(x_train, y_train_propagated)
# print(log_reg.score(x_test, y_test))

percentile_closest = 99

x_cluster_dist = x_digits_dist[np.arange(len(x_train)), kmeans.labels_]
for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = x_cluster_dist[in_cluster]
    cutoff_dist = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = (x_cluster_dist > cutoff_dist)
    x_cluster_dist[in_cluster & above_cutoff] = -1

partially_propagated = (x_cluster_dist != -1)
x_train_partially_propagated = x_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]

log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(x_train_partially_propagated, y_train_partially_propagated)
# print(log_reg.score(x_test, y_test))
# print((y_train_partially_propagated == y_train[partially_propagated]).mean())

'''-----------------------------DBSCAN---------------------------------------'''

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

x, y = make_moons(n_samples=1000, noise=0.05)
dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(x)

# print(dbscan.labels_)
# print(dbscan.core_sample_indices_)
# print(dbscan.components_)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])

x_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])
# print(knn.predict(x_new))
# print(knn.predict_proba(x_new))

y_dist, y_pred_idx = knn.kneighbors(x_new, n_neighbors=1)
y_pred = dbscan.labels_[dbscan.core_sample_indices_][y_pred_idx]
y_pred[y_dist > 0.2] = -1
# print(y_pred.ravel()) # [-1 3 6 -1]

'''----------------------------Gaussian Mixture-------------------------------'''

from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=3, n_init=10)
gm.fit(x)

# print(gm.weights_)
# print(gm.means_)
# print(gm.covariances_)

# print(gm.converged_) # True
# print(gm.n_iter_) # It took 16 iterations to converge

# print(gm.predict(x))
# print(gm.predict_proba(x).round(3))

x_new, y_new = gm.sample(6)
# print(x_new)
# print()
# print(y_new)
# print()
# print(gm.score_samples(x).round(2))

#----------------------Gaussian Mixtures for Anomaly Detection----------------------

densities = gm.score_samples(x)
density_threshold = np.percentile(densities, 2)
anomalies = x[densities < density_threshold]

#-----------------------Bayesian Gaussian Misxture Models--------------------------

# print(gm.bic(x))

from sklearn.mixture import BayesianGaussianMixture

bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
bgm.fit(x)
# print(bgm.weights_.round(2))

'''-----------------------------Isolation Forest---------------------------------- '''
from sklearn.ensemble import IsolationForest

x = [[-1.1], [0.3], [0.5], [100]]
iso_forest = IsolationForest(random_state=42)
clf = iso_forest.fit(x)
# print(clf.predict([[0.1], [0], [90]])) # [1, 1, -1]

