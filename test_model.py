import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class data:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train


@pytest.fixture
def train_data():
    df = pd.read_csv("census.csv")
    X_train, y_train, *_ = process_data(
        df, categorical_features=cat_features, label="salary", training=True
    )
    return data(X_train, y_train)


@pytest.fixture
def mock_model():
    X, y = make_classification(n_samples=1000, n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)
    return clf


def test_train_model(train_data):
    X_train = train_data.X_train
    y_train = train_data.y_train
    rfc = train_model(X_train, y_train)
    rfc_mock = RandomForestClassifier()
    assert type(rfc) == type(rfc_mock)
    assert rfc is not rfc_mock


def test_compute_model_metrics():
    y_mock = np.array([0, 0, 1])
    y_mock_ = 1 - y_mock

    same_precision, same_recall, same_fbeta = compute_model_metrics(y_mock, y_mock)

    assert same_precision == 1
    assert same_recall == 1
    assert same_fbeta == 1

    different_precison, different_recall, different_fbeta = compute_model_metrics(y_mock, y_mock_)

    assert different_precison == 0
    assert different_recall == 0
    assert different_fbeta == 0


def test_inference(mock_model, train_data):
    X = [[0, 0, 0, 0]]
    pred = inference(mock_model, X)
    assert len(pred) == len(X)