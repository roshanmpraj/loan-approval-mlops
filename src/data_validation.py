import pandas as pd


# =========================================================
# Configuration
# =========================================================

DATA_PATH = "data/loan_approval.csv"


# Columns expected by our ML pipeline
EXPECTED_COLUMNS = [
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


# =========================================================
# Load data
# =========================================================

def load_data():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    return df


# =========================================================
# Validate schema
# =========================================================

def validate_schema(df):
    print("\nValidating schema...")

    actual_columns = list(df.columns)

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in actual_columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print("✓ Schema validation passed")


# =========================================================
# Validate target
# =========================================================

def validate_target(df):
    print("\nValidating target...")

    allowed_values = {"Y", "N"}

    actual_values = set(
        df["Loan_Status"]
        .dropna()
        .unique()
    )

    invalid_values = actual_values - allowed_values

    if invalid_values:
        raise ValueError(
            f"Invalid Loan_Status values: {invalid_values}"
        )

    print("✓ Target validation passed")


# =========================================================
# Check missing values
# =========================================================

def check_missing_values(df):
    print("\nChecking missing values...")

    missing = df.isnull().sum()

    print(missing)


# =========================================================
# Check duplicate rows
# =========================================================

def check_duplicates(df):
    print("\nChecking duplicates...")

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate rows: {duplicate_count}")


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    df = load_data()

    validate_schema(df)

    validate_target(df)

    check_missing_values(df)

    check_duplicates(df)

    print("\n✓ Data validation completed successfully")