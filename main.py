from fastapi import FastAPI
import numpy as np
import joblib
from pydantic import BaseModel

app = FastAPI()


class FlightFeatures(BaseModel):
    airline: int
    flight: int
    source_city: int
    departure_time: int
    stops: int
    arrival_time: int
    destination_city: int
    class_: int
    duration: float
    days_left: int

    class Config:
        fields = {'class_': 'class'}

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/predict")
def predict_price(features: FlightFeatures):
    model = joblib.load('random_forest_model.pkl')
    features_dict = features.model_dump()
    features_array = np.array(list(features_dict.values())).reshape(1, -1)
    prediction = model.predict(features_array)
    return {"prediction": prediction[0]}



