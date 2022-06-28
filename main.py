import pickle

from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd

import ml.model
from ml.data import process_data
from ml.model import train_model

import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()

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


class InferenceInput(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        }


@app.get("/")
async def greeting():
    return {"message": "Hello World"}


@app.post("/inference/")
async def inference(input: InferenceInput):
    X_inference = pd.Series(input.dict(by_alias=True)).to_frame().T
    model = pickle.load(open('rfc_model.sav', 'rb'))
    encoder = pickle.load(open('encoder.sav', 'rb'))
    X_inference, *_ = process_data(X_inference,
                                   categorical_features=cat_features,
                                   training=False,
                                   encoder=encoder
                                   )
    pred = ml.model.inference(model, X_inference)
    return {"prediction": str(pred)}
