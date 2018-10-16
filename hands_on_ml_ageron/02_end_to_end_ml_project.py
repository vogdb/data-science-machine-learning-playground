import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import randint

DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = os.path.join('datasets', 'housing')
HOUSING_URL = DOWNLOAD_ROOT + 'datasets/housing/housing.tgz'


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


def stratified_split(X):
    X['income_cat'] = np.ceil(X['median_income'] / 1.5)
    X['income_cat'].where(X['income_cat'] < 5, 5.0, inplace=True)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_index, test_index = next(split.split(X, X['income_cat']))
    X.drop('income_cat', axis=1, inplace=True)
    return train_index, test_index


class CombinedAttrsAdder(BaseEstimator, TransformerMixin):
    def __init__(self, rooms_ix=3, bedrooms_ix=4, population_ix=5, household_ix=6, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.rooms_ix = rooms_ix
        self.bedrooms_ix = bedrooms_ix
        self.population_ix = population_ix
        self.household_ix = household_ix
        # ['rooms_per_household', 'population_per_household', 'bedrooms_per_room']

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.household_ix]
        population_per_household = X[:, self.population_ix] / X[:, self.household_ix]
        transformed = np.c_[X, rooms_per_household, population_per_household]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            transformed = np.c_[transformed, bedrooms_per_room]
        return transformed


def display_scores(scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard deviation:', scores.std())


def cross_val_score_example(X, y):
    print('CROSS VALIDATION SCORE of default params models: ')
    model_list = {
        'lin_reg': LinearRegression(),
        'tree_reg': DecisionTreeRegressor(random_state=42),
        'rnd_forest_reg': RandomForestRegressor(random_state=42, n_estimators=10),
    }
    for model_name, model in model_list.items():
        scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=10)
        rmse = np.sqrt(-scores)
        print('"{}" scores:'.format(model_name))
        display_scores(rmse)
        print('---------------------------------------------------------')


def grid_search_example(X, y):
    estimator = RandomForestRegressor(random_state=42)
    print('Grid Search CV of "{}" model:'.format(estimator.__class__.__name__))
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
    grid_search = GridSearchCV(estimator, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X, y)

    print(grid_search.best_params_)
    print(grid_search.best_estimator_)
    print(pd.DataFrame(grid_search.cv_results_).describe())

    # TODO would be interesting to inject at the end of `prepare` pipeline a restoration of column names to X
    # the below way is very shaky
    feature_importances = grid_search.best_estimator_.feature_importances_
    extra_attr_list = ['rooms_per_hhold', 'pop_per_hhold', 'bedrooms_per_room']
    cat_encoder = prepare_pipeline.named_transformers_['cat']
    one_hot_attr_list = list(cat_encoder.categories_[0])
    attrs = num_attr_list + extra_attr_list + one_hot_attr_list
    print(sorted(zip(feature_importances, attrs), reverse=True))


def randomized_search_cv_example(X, y):
    predictor = RandomForestRegressor(random_state=42)
    print('Random Search CV of "{}" model:'.format(predictor.__class__.__name__))

    param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }
    rnd_search = RandomizedSearchCV(
        predictor, param_distributions=param_distribs,
        n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42
    )
    rnd_search.fit(X, y)
    print(rnd_search.best_params_)


def single_pipeline_example(X, y):
    print('Full pipeline example')
    svm = SVR(gamma=0.26497040005002437, C=157055.10989448498)
    full_pipeline = Pipeline([
        ('prepare', prepare_pipeline),
        ('train', svm)
    ])
    full_pipeline.fit(X, y)
    scores = full_pipeline.predict(X.iloc[:5])
    print('predict scores:\n{}'.format(list(map(int, scores))))
    print('actual scores: \n{}'.format(list(map(int, y.iloc[:5]))))


# load dataset
# fetch_housing_data()
housing = load_housing_data()
X = housing.drop('median_house_value', axis=1)
y = housing['median_house_value'].copy()

# split on training and test datasets
train_index, test_index = stratified_split(X)
X_train, y_train = X.iloc[train_index], y.iloc[train_index]
X_test, y_test = X.iloc[test_index], y.iloc[test_index]

# group attributes names
cat_attr_list = ['ocean_proximity']
num_attr_list = list(X)
for cat_attr in cat_attr_list:
    num_attr_list.remove(cat_attr)

# prepare data
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attrs_adder', CombinedAttrsAdder()),
    ('std_scaler', StandardScaler()),
])

prepare_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attr_list),
    ('cat', OneHotEncoder(), cat_attr_list),
])

X_train_prep = prepare_pipeline.fit_transform(X_train)

# cross_val_score_example(X_train_prep, y_train)
# grid_search_example(X_train_prep, y_train)
randomized_search_cv_example(X_train_prep, y_train)
# single_pipeline_example(X_train, y_train)
