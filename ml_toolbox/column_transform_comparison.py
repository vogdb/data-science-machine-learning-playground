# Data is assumed to be taken from https://www.kaggle.com/c/titanic
import numpy as np
import os
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Imputer


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series(
            [X[c].value_counts().index[0] for c in X],
            index=X.columns
        )
        return self

    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


def new_school_prepare(df):
    age_transformer = Pipeline([
        ('fill na', SimpleImputer(strategy='median')),
        # ('scale', StandardScaler()),
    ])

    embarked_transformer = Pipeline([
        ('fill na', MostFrequentImputer()),
        ('onehot', OneHotEncoder(sparse=False)),
    ])

    prepare_pipeline = ColumnTransformer([
        ('age', age_transformer, ['Age']),
        ('pass', 'passthrough', ['SibSp', 'Parch', 'Fare']),
        # ('num', StandardScaler(), ['SibSp', 'Parch', 'Fare']),
        ('cat', OneHotEncoder(sparse=False), ['Pclass', 'Sex']),
        ('embarked', embarked_transformer, ['Embarked']),
    ], remainder='drop')

    return prepare_pipeline.fit_transform(df)


def old_school_prepare(df):
    class DataFrameSelector(BaseEstimator, TransformerMixin):
        def __init__(self, attribute_names):
            self.attribute_names = attribute_names

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X[self.attribute_names]

    num_pipeline = Pipeline([
        ('select_numeric', DataFrameSelector(['Age', 'SibSp', 'Parch', 'Fare'])),
        ('imputer', Imputer(strategy='median')),
    ])

    cat_pipeline = Pipeline([
        ('select_cat', DataFrameSelector(['Pclass', 'Sex', 'Embarked'])),
        ('imputer', MostFrequentImputer()),
        ('cat_encoder', OneHotEncoder(sparse=False)),
    ])

    prepare_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline),
    ])

    return prepare_pipeline.fit_transform(df)


def compare_prepare_methods():
    def about_X(X):
        print(type(X))
        print(X.shape)
        print(X[:5])

    X_new = new_school_prepare(df)
    X_old = old_school_prepare(df)
    print('X new:')
    about_X(X_new)
    print('X old:')
    about_X(X_old)
    print(np.allclose(X_old, X_new))


df = pd.read_csv(os.path.join('dataset', 'train.csv'))
compare_prepare_methods()
