from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Create FastAPI app
app = FastAPI(title="Diabetes Prediction API")

# Load trained model
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

# Input schema (replaces Streamlit sidebar inputs)
class PatientInput(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: int

# Home route
@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running"}

# Prediction route
@app.post("/predict")
def predict_diabetes(data: PatientInput):

    features = np.array([[  
        data.pregnancies,
        data.glucose,
        data.blood_pressure,
        data.skin_thickness,
        data.insulin,
        data.bmi,
        data.diabetes_pedigree_function,
        data.age
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1] * 100

    return {
        "prediction": "Likely Diabetic" if prediction == 1 else "Likely Not Diabetic",
        "risk_percentage": round(probability, 2)
    }