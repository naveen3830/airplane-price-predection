from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

app = FastAPI()

# Input schema using Pydantic
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
        fields = {"class_": "class"}  # map 'class_' to 'class' for JSON input


# Load model on startup and store in app state
@app.on_event("startup")
def load_model():
    model_path = hf_hub_download(
        repo_id="naveen3830/my-ml-model",
        filename="random_forest_model.pkl"
    )
    app.state.model = joblib.load(model_path)


@app.get("/")
def root():
    return {"message": "Flight price prediction API is live!"}


@app.post("/predict")
def predict_price(features: FlightFeatures):
    model = app.state.model
    input_array = np.array(list(features.model_dump().values())).reshape(1, -1)
    prediction = model.predict(input_array)
    return {"prediction": float(prediction[0])}
