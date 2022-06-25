import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

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
    def __init__(self, X, y):
        self.X = X
        self.y = y


@pytest.fixture
def mock_train_data():
    X, y = make_classification(n_samples=1000, n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)
    return data(X, y)


@pytest.fixture
def mock_model(mock_train_data):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(mock_train_data.X, mock_train_data.y)
    return clf


def test_train_model(mock_train_data,mock_model):
    rfc = train_model(mock_train_data.X, mock_train_data.y)
    assert type(rfc) == type(mock_model)
    assert rfc is not mock_model


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


def test_inference(mock_model, mock_train_data):
    mock_train_data.X = [[0, 0, 0, 0]]
    pred = inference(mock_model, mock_train_data.X)
    assert len(pred) == len(mock_train_data.X)