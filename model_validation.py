import json
import pickle

import pandas as pd

from ml.data import process_data
from ml.model import inference, compute_model_metrics
from train_model import cat_features


def get_model():
    model = pickle.load(open('rfc_model.sav', 'rb'))
    encoder = pickle.load(open('encoder.sav', 'rb'))
    lb = pickle.load(open('lb.sav', 'rb'))
    return model, encoder, lb


def val_slice(data, model, encoder, lb, feature):
    slice_validation = dict()
    for slice in data[feature].unique():
        slice_data = data.loc[data[feature] == slice]
        X, y, *_ = process_data(
            slice_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
        )
        preds = inference(model, X)
        precision, recall, fbeta = compute_model_metrics(y, preds)
        slice_validation[slice] = {"precision": precision}
        slice_validation[slice]["recall"] = recall
        slice_validation[slice]["fbeta"] = fbeta
    return slice_validation


if __name__ == '__main__':
    data = pd.read_csv("census.csv")
    model, encoder, lb = get_model()
    feature = "education"
    slice_validation = val_slice(data, model, encoder, lb, feature)
    with open('slice_output.txt', 'w') as convert_file:
        convert_file.write(json.dumps(slice_validation))
