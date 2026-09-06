import pandas as pd

DATA_PATH = "data/loan_approval.csv"


def test_data_file_exists():
    df = pd.read_csv(DATA_PATH)

    assert not df.empty


def test_expected_columns_exist():
    df = pd.read_csv(DATA_PATH)

    expected_columns = [
        "Loan_ID",
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Property_Area",
        "Loan_Status",
    ]

    assert list(df.columns) == expected_columns


def test_target_column_exists():
    df = pd.read_csv(DATA_PATH)

    assert "Loan_Status" in df.columns


def test_target_values_are_valid():
    df = pd.read_csv(DATA_PATH)

    assert set(df["Loan_Status"].dropna().unique()) <= {"Y", "N"}
