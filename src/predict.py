import joblib
import pandas as pd


# =========================================================
# Configuration
# =========================================================

MODEL_PATH = "models/loan_approval_model.pkl"


# =========================================================
# Load Model
# =========================================================

def load_model():

    print("Loading trained model...")

    model = joblib.load(MODEL_PATH)

    print("✓ Model loaded successfully")

    return model


# =========================================================
# Make Prediction
# =========================================================

def predict_loan(model, applicant_data):

    # Convert applicant data into a DataFrame
    input_data = pd.DataFrame([applicant_data])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get prediction probability
    probabilities = model.predict_proba(input_data)[0]

    # Find probability of Y = Approved
    approved_probability = probabilities[
        list(model.classes_).index("Y")
    ]

    return prediction, approved_probability


# =========================================================
# Main Execution
# =========================================================

if __name__ == "__main__":

    # 1. Load saved model
    model = load_model()

    # 2. Example applicant
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

    # 3. Make prediction
    prediction, approved_probability = predict_loan(
        model,
        applicant
    )

    # 4. Display result
    print("\n===================================")
    print("LOAN PREDICTION")
    print("===================================")

    if prediction == "Y":
        print("Prediction: APPROVED")
    else:
        print("Prediction: NOT APPROVED")

    print(
        f"Approval Probability: "
        f"{approved_probability:.2%}"
    )

    print("===================================")
