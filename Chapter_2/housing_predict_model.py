'''
This is a program that predicts the median housing price.
'''

import sklearn
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()

# Taking a quick look at the data
# print(housing.head())
# print(housing.info())
# print(housing["ocean_proximity"].value_counts())
# print(housing.describe())

# extra code – code to save the figures as high-res PNGs for the book

IMAGES_PATH = Path() / "images" / "end_to_end_project"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Display some plots for our data  
def display_hist():
    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14, titlesize=14)
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    housing.hist(bins=50, figsize=(12, 8))
    save_fig("attribute_histogram_plots")  # extra code
    plt.show()

# Create the test and train set based on an 80/20 split
def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

from zlib import crc32

# Checking if each instance will be placed in the test set
def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio *2**32

# Ensuring that the same instances go to the same set each time
def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# train_set, test_set = shuffle_and_split_data(housing, 0.2)
# print(len(train_set)) # 16512
# print(len(test_set)) # 4128

from sklearn.model_selection import train_test_split

housing_with_id = housing.reset_index()  # adds an `index` column
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
# train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# test_set["total_bedrooms"].isnull.sum()

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

# housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
# plt.xlabel("Income category")
# plt.ylabel("Number of districts")
# save_fig("housing_income_cat_bar_plot")
# plt.show()

# extra code – shows how to compute the 10.7% proba of getting a bad sample
from scipy.stats import binom

def get_bad_sample_prob():
    sample_size = 1000
    ratio_female = 0.511
    proba_too_small = binom(sample_size, ratio_female).cdf(485 - 1)
    proba_too_large = 1 - binom(sample_size, ratio_female).cdf(535)
    print(proba_too_small + proba_too_large)

# extra code – shows another way to estimate the probability of bad sample

np.random.seed(42)

def get_bad_sample_prob_alt(sample_size, ratio_female):
    samples = (np.random.rand(100_000, sample_size) < ratio_female).sum(axis=1)
    ((samples < 485) | (samples > 535)).mean()

from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

def income_cat_proportions(data):
    return (data["income_cat"].value_counts() / len(data))

def print_props():
    compare_props = pd.DataFrame({
        "Overall %": income_cat_proportions(housing),
        "Stratified %": income_cat_proportions(strat_test_set),
        "Random %": income_cat_proportions(test_set),
    }).sort_index()
    compare_props.index.name = "Income Category"
    compare_props["Strat. Error %"] = (compare_props["Stratified %"] /
                                    compare_props["Overall %"] - 1)
    compare_props["Rand. Error %"] = (compare_props["Random %"] /
                                    compare_props["Overall %"] - 1)
    print((compare_props * 100).round(2))

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()

# Scatterplot with longitude and latitude
# housing.plot(kind="scatter", x="longitude", y="latitude", grid=True)
# # plt.show()

# More detailed scatterplot
def display_long_lat_scatterplot():
    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
    # plt.show()

    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
                s=housing["population"] / 100, label="population",
                c="median_house_value", cmap="jet", colorbar=True,
                legend=True, sharex=False, figsize=(10, 7))
    save_fig("housing_prices_scatterplot")  # extra code
    plt.show()

# extra code – this cell generates the first figure in the chapter

# Download the California image
def get_california_image():
    filename = "california.png"
    if not (IMAGES_PATH / filename).is_file():
        homl3_root = "https://github.com/ageron/handson-ml3/raw/main/"
        url = homl3_root + "images/end_to_end_project/" + filename
        print("Downloading", filename)
        urllib.request.urlretrieve(url, IMAGES_PATH / filename)

    housing_renamed = housing.rename(columns={
        "latitude": "Latitude", "longitude": "Longitude",
        "population": "Population",
        "median_house_value": "Median house value (ᴜsᴅ)"})
    housing_renamed.plot(
                kind="scatter", x="Longitude", y="Latitude",
                s=housing_renamed["Population"] / 100, label="Population",
                c="Median house value (ᴜsᴅ)", cmap="jet", colorbar=True,
                legend=True, sharex=False, figsize=(10, 7))

    california_img = plt.imread(IMAGES_PATH / filename)
    axis = -124.55, -113.95, 32.45, 42.05
    plt.axis(axis)
    plt.imshow(california_img, extent=axis)

    save_fig("california_housing_prices_plot")
    plt.show()

# Display a correlation matrix
corr_matrix = housing.corr(numeric_only=True)

corr_matrix["median_house_value"].sort_values(ascending=False)

# Display a scatterplot matrix
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
# save_fig("scatter_matrix_plot")

housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1, grid=True)
save_fig("income_vs_house_value_scatterplot")  # extra code

# Here, we will experiment with different attributes
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]
corr_matrix = housing.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)

''' Now, we will prepare the data for ML algorithms '''

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Clean the data so we don't have missing values: we replace any NA values with the median

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
# print(imputer.statistics_)
# print(housing_num.median().values)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index = housing_num.index) # Add columns and index

# Drop outliers with isolation forests
from sklearn.ensemble import IsolationForest

isolation_forest = IsolationForest(random_state=42)
outlier_pred = isolation_forest.fit_predict(X)
housing = housing.iloc[outlier_pred == 1]
housing_labels = housing_labels.iloc[outlier_pred == 1]

'''Handling text and categorical attributes'''

housing_cat = housing[["ocean_proximity"]]
# print(housing_cat.head(8))

# Convert categories from text to numbers
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
# print(housing_cat_encoded[:8])
# print(ordinal_encoder.categories_)

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# print(housing_cat_1hot.toarray())

# Get the dummy attributes
df_test = pd.DataFrame({"ocean_proximity": ["INLAND", "NEAR BAY"]})
# print(pd.get_dummies(df_test))

'''Feature Scaling and Transformation'''

# Normalization
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=(-1,1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

# Standardization
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

# Creating a Gaussian RBF feature
from sklearn.metrics.pairwise import rbf_kernel

age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)

# Transform the target values
from sklearn.linear_model import LinearRegression

target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())
model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)
some_new_data = housing[["median_income"]].iloc[:5] # let's pretend this is new data
scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)
# print(predictions)

# Easier Method:
from sklearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)
# print(predictions)

'''Custom Transformers'''

from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])

rbf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[[35.]], gamma=0.1))
age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])

sf_coords = 37.7749, -122.41
sf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[sf_coords], gamma=0.1))
sf_simil = sf_transformer.transform(housing[["latitude", "longitude"]])

# We can build a custom scaler that behaves like a standard scaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):  # no *args or **kwargs!
        self.with_mean = with_mean

    def fit(self, X, y=None):  # y is required even though we don't use it
        X = check_array(X)  # checks that X is an array with finite float values
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]  # every estimator stores this in fit()
        return self  # always return self!

    def transform(self, X):
        check_is_fitted(self)  # looks for learned attributes (with trailing _)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_
    
# We can also build a custom transformer that uses k-means clustering
import warnings
from sklearn.cluster import KMeans

warnings.simplefilter(action='ignore', category=FutureWarning)

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
    

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]],
                                           sample_weight=housing_labels)

'''Transformation Pipelines'''

# Small pipeline for numerical attributes which will impute and then scale
from sklearn.pipeline import Pipeline, make_pipeline

sklearn.set_config(display="diagram")
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler())
])
# We can also make a pipeline this way:
num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

housing_num_prepared = num_pipeline.fit_transform(housing_num)
# print(housing_num_prepared[:2].round(2))
# Making it nicer to see
df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared, columns=num_pipeline.get_feature_names_out(),
    index=housing_num.index
)
# print(df_housing_num_prepared)

# We can make a single pipeline for numerical attributes and categorical attributes
from sklearn.compose import ColumnTransformer

num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
               "total_bedrooms", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)
preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

from sklearn.compose import make_column_selector, make_column_transformer

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)

housing_prepared = preprocessing.fit_transform(housing) # our preprocessing pipeline
df_housing_prepared = pd.DataFrame(
    housing_prepared, columns=preprocessing.get_feature_names_out(),
    index=housing.index
)
# print(df_housing_prepared)

# Entire process summarized and organized
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())
preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)  # one column remaining: housing_median_age

housing_prepared = preprocessing.fit_transform(housing)
# print(housing_prepared.shape)
# print(preprocessing.get_feature_names_out())

'''Train and Evaluate on the Training Set'''

# Make a pipeline for a linear regression model
lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)

# Using the training set
housing_predictions = lin_reg.predict(housing)
# Compare predicted and actual values
# print(housing_predictions[:5].round(-2)) # rounded to the nearest hundred
# print(housing_labels.iloc[:5].values)

from sklearn.metrics import mean_squared_error

lin_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
# print(lin_rmse) # We are off by $65,778.48 typically, so our model is underfitting

# We could use a decision tree regressor to check complex nonlinear relationships
from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)
housing_predictions = tree_reg.predict(housing)
tree_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
# print(tree_rmse) # outputs 0.0, which indicates no error, which may imply overfitting

'''Better Evaluation using Cross-Validation'''

from sklearn.model_selection import cross_val_score

def print_tree_score():
    # This value is negative b/c the score gives us the utilty function score, not the cost function score
    tree_rmses = -cross_val_score(tree_reg, housing, housing_labels,
                                scoring ="neg_root_mean_squared_error", cv=10)
    print(pd.Series(tree_rmses).describe())

# Now, let's try a random forest regressor

from sklearn.ensemble import RandomForestRegressor

def perform_rnd_forest():
    forest_reg = make_pipeline(preprocessing,
                            RandomForestRegressor(random_state=42))
    forest_rmses = -cross_val_score(forest_reg, housing, housing_labels,
                                    scoring="neg_root_mean_squared_error", cv=10)

    print(pd.Series(forest_rmses).describe())

'''Fine-Tuning the Model'''

# Grid Search
from sklearn.model_selection import GridSearchCV

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42)),
])

def perform_grid_search():
    param_grid = [
        {'preprocessing__geo__n_clusters': [5, 8, 10],
        'random_forest__max_features': [4, 6, 8]},
        {'preprocessing__geo__n_clusters': [10, 15],
        'random_forest__max_features': [6, 8, 10]},
    ]
    grid_search = GridSearchCV(full_pipeline, param_grid, cv=3, scoring='neg_root_mean_squared_error')
    grid_search.fit(housing, housing_labels)
    # print(grid_search.best_params_)
    # print(grid_search.best_estimator_)
    cv_res = pd.DataFrame(grid_search.cv_results_)
    cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)

# Randomized Search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

def perform_rnd_search():
    param_distribs = {'preprocessing__geo__n_clusters': randint(low=3, high=50),
                    'random_forest__max_features': randint(low=2, high=20)}
    rnd_search = RandomizedSearchCV(
        full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
        scoring='neg_root_mean_squared_error', random_state=42
    )
    rnd_search.fit(housing, housing_labels)

'''Analyzing the Best Models and Their Errors'''

rnd_search = None
perform_rnd_search()
final_model = rnd_search.best_estimator_
feature_importances = final_model["random_forest"].feature_importances_
print(feature_importances.round(2)) # prints an array of the importance value of each attribute
print(sorted(zip(
    feature_importances,
    final_model["preprocessing"].get_feature_names_out()),
    reverse=True
))

'''Evaluate the System on the Test Set'''

X_test = strat_test_set.drop("median_house_value", axis=1)
Y_test = strat_test_set["median_house_value"].copy()
final_predictions = final_model.predict(X_test)
final_rmse = mean_squared_error(Y_test, final_predictions, squared=False)
print(final_rmse)

from scipy import stats

# Get the 95% confidence interval
confidence = 0.95
squared_errors = (final_predictions - Y_test) ** 2
print(np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)
                         )))


'''Launch, Monitor, and Maintain the System'''

import joblib

joblib.dump(final_model, "my_california_housing_model.pk1")
# We can load the model like this:
final_model_reloaded = joblib.load("my_california_housing_model.pk1")
# Make further predictions with new data...