# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

import pickle

# Add the necessary imports for the starter code.

# Add code to load in the data.

data = pd.read_csv("census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, *_ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, lb=lb, encoder=encoder
)

model = train_model(X_train=X_train, y_train=y_train)

preds = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, preds)

print("precision: " + str(precision))

print("recall: " + str(recall))

print("fbeta: " + str(fbeta))

X, y, encoder, lb = process_data(
    data, categorical_features=cat_features, label="salary", training=True
)

model = train_model(X_train=X_train, y_train=y_train)

pickle.dump(model, open('rfc_model.sav', 'wb'))

pickle.dump(encoder, open('encoder.sav', 'wb'))

