import pytest
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.mlproject.pipeline.prediction_pipeline import PredictionPipeline
from flask_app import app

def test_prediction_from_dict():
    pipeline = PredictionPipeline()
    sample = {
        "continent": "Asia",
        "education_of_employee": "Doctorate",
        "has_job_experience": "Y",
        "requires_job_training": "N",
        "no_of_employees": 1337,
        "yr_of_estab": 1989,
        "region_of_employment": "South",
        "prevailing_wage": 199777.59,
        "unit_of_wage": "Year",
        "full_time_position": "Y"
    }

    prediction = pipeline.predict(sample)
    assert prediction is not None
    assert len(prediction) == 1


@pytest.fixture
def client():
    app.testing = True
    return app.test_client()

def test_index_prediction(client):
    response = client.post("/", data={
        "continent": "Asia",
        "education_of_employee": "Doctorate",
        "has_job_experience": "Y",
        "requires_job_training": "N",
        "no_of_employees": 1337,
        "yr_of_estab": 1989,
        "region_of_employment": "South",
        "prevailing_wage": 199777.59,
        "unit_of_wage": "Year",
        "full_time_position": "Y"
    })
    assert response.status_code == 200
    assert b"Predicted Case Status" in response.data