'''This program showcases dimensionality reduction algorithms'''

'''----------------------------------PCA--------------------------------------'''

import numpy as np
from sklearn.decomposition import PCA

def dim_reduce(data_3d):
    x = data_3d # some small 3D dataset
    x_centered = x - x.mean(axis=0) # PCA assumes the data is centered around the origin
    U, s, Vt = np.linalg.svd(x_centered)
    c1 = Vt[0]
    c2 = Vt[1]

    # Projecting the training set onto the plane, we perform matrix multiplication
    W2 = Vt[:2].T
    X2D = x_centered @ W2

    # Using Scikit-Learn
    pca = PCA(n_components=2)
    X2D = pca.fit_transform(x)
    print(pca.explained_variance_ratio_)

'''-------------------Choosing the Right Number of Dimensions---------------------'''

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')
x_train, y_train = mnist.data[:60_000], mnist.target[:60_000]
x_test, y_test = mnist.data[60_000:], mnist.target[60_000:]

pca = PCA()
pca.fit(x_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1 # d equals 154

pca = PCA(n_components=0.95)
x_reduced = pca.fit_transform(x_train)
# print(pca.n_components_) # 154

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline

clf = make_pipeline(PCA(random_state=42),
                    RandomForestClassifier(random_state=42))

param_distrib = {
    "pca__n_components": np.arange(10, 80),
    "randomforestclassifier__n_estimators": np.arange(50, 500)
}
rnd_search = RandomizedSearchCV(clf, param_distrib, n_iter=10, cv=3,
                                random_state=42)
rnd_search.fit(x_train[:1000], y_train[:1000])
# print(rnd_search.best_params_) # {'randomforestclassifier__n_estimators': 465, 'pca__n_components': 23}

x_recovered = pca.inverse_transform(x_reduced)

# Randomized PCA
rnd_pca = PCA(n_components=154, svd_solver="randomized", random_state=42)
x_reduced = rnd_pca.fit_transform(x_train)

'''-------------------------------Incremental PCA------------------------------------'''

from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for x_batch in np.array_split(x_train, n_batches):
    inc_pca.partial_fit(x_batch)

x_reduced = inc_pca.transform(x_train)

def build_mmap(filename):
    x_mmap = np.memmap(filename, dtype='float32', mode='write', shape=x_train.shape)
    x_mmap[:] = x_train # could be a loop instead, saving the data chunk by chunk
    x_mmap.flush()

    x_mmap = np.memmap(filename, dtype="float32", mode="readonly").reshape(-1, 784)
    batch_size = x_mmap.shape[0]
    inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
    inc_pca.fit(x_mmap)

'''--------------------------------Random Projection--------------------------------'''

from sklearn.random_projection import johnson_lindenstrauss_min_dim

m, eps = 5_000, 0.1
d = johnson_lindenstrauss_min_dim(m, eps=eps)
# print(d) # 7300

n = 20_000
np.random.seed(42)
P = np.random.randn(d, n) / np.sqrt(d) # std dev = square root of variance
x = np.random.randn(m, n) # generate a fake dataset
x_reduced = x @ P.T

from sklearn.random_projection import GaussianRandomProjection

gaussian_rnd_proj = GaussianRandomProjection(eps=eps, random_state=42)
x_reduced = gaussian_rnd_proj.fit_transform(x) # same result as above

# Perform inverse transformation
components_pinv = np.linalg.pinv(gaussian_rnd_proj.components_)
x_recovered = x_reduced @ components_pinv.T

'''-----------------------------------------LLE----------------------------------------'''

from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

x_swiss , t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
x_unrolled = lle.fit_transform(x_swiss)