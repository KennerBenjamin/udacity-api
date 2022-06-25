from fastapi.testclient import TestClient

from inference_api import app, InferenceInput

client = TestClient(app)

low_salary_example = {
    'age': 39,
    'workclass': 'State-gov',
    'fnlgt': 77516,
    'education': 'Bachelors',
    'education-num': 13,
    'marital-status': 'Never-married',
    'occupation': 'Adm-clerical',
    'relationship': 'Not-in-family',
    'race': 'White',
    'sex': 'Male',
    'capital-gain': 2174,
    'capital-loss': 0,
    'hours-per-week': 40,
    'native-country': 'United-States'
}

high_salary_example = {
    "age": 42,
    "workclass": "Private",
    "fnlgt": 159449,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 5178,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Hello World"}


def test_low_salary():
    r = client.post("/inference/", json=low_salary_example)
    assert r.status_code == 200
    assert r.json() == {"prediction": "[0]"}


def test_high_salary():
    r = client.post("/inference/", json=high_salary_example)
    assert r.status_code == 200
    assert r.json() == {"prediction": "[1]"}
