import joblib
import pandas as pd

MODEL_PATH = "models/loan_approval_model.pkl"


def test_model_prediction():
    model = joblib.load(MODEL_PATH)

    applicant = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 2000,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Property_Area": "Urban",
    }

    applicant_df = pd.DataFrame([applicant])

    prediction = model.predict(applicant_df)[0]

    assert prediction in ["Y", "N"]