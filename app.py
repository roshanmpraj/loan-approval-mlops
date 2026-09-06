from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# Create FastAPI application
app = FastAPI(
    title="Loan Approval Prediction API",
    version="1.0.0"
)


# Load trained ML pipeline
model = joblib.load("models/loan_approval_model.pkl")


# Input data schema
class LoanApplication(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str


# Health check
@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "loan-approval-api"
    }


# Prediction endpoint
@app.post("/predict")
def predict(application: LoanApplication):

    # Convert request to DataFrame
    data = pd.DataFrame([application.model_dump()])

    # Make prediction
    prediction = model.predict(data)[0]

    # Get probability of approval
    probability = model.predict_proba(data)[0][1] * 100

    return {
        "prediction": (
            "APPROVED"
            if prediction == "Y"
            else "REJECTED"
        ),
        "approval_probability": round(
            float(probability),
            2
        )
    }