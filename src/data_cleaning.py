import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# =========================================================
# Configuration
# =========================================================

DATA_PATH = "data/loan_approval.csv"


# =========================================================
# Feature columns
# =========================================================

NUMERICAL_COLUMNS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

CATEGORICAL_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
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
# Remove unnecessary columns
# =========================================================

def remove_unnecessary_columns(df):
    print("\nRemoving unnecessary columns...")

    df = df.drop(columns=["Loan_ID"])

    print("✓ Loan_ID removed")

    return df


# =========================================================
# Check missing values
# =========================================================

def check_missing_values(df):
    print("\nChecking missing values...")

    missing = df.isnull().sum()

    print(missing)

    total_missing = missing.sum()

    print(f"\nTotal missing values: {total_missing}")


# =========================================================
# Split features and target
# =========================================================

def split_data(df):
    print("\nSplitting features and target...")

    X = df.drop(columns=["Loan_Status"])

    y = df["Loan_Status"]

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    return X, y


# =========================================================
# Train / Test Split
# =========================================================

def create_train_test_split(X, y):
    print("\nCreating train/test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test


# =========================================================
# Numerical preprocessing
# =========================================================

def create_numerical_pipeline():

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    return numerical_pipeline


# =========================================================
# Categorical preprocessing
# =========================================================

def create_categorical_pipeline():

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    return categorical_pipeline


# =========================================================
# Complete preprocessing pipeline
# =========================================================

def create_preprocessing_pipeline():

    numerical_pipeline = create_numerical_pipeline()

    categorical_pipeline = create_categorical_pipeline()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_COLUMNS
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_COLUMNS
            )
        ]
    )

    return preprocessor


# =========================================================
# Fit and transform preprocessing
# =========================================================

def preprocess_data(X_train, X_test):

    print("\nCreating preprocessing pipeline...")

    preprocessor = create_preprocessing_pipeline()

    print("Fitting preprocessor on training data...")

    X_train_processed = preprocessor.fit_transform(X_train)

    print("Transforming test data...")

    X_test_processed = preprocessor.transform(X_test)

    print(
        f"Processed X_train shape: "
        f"{X_train_processed.shape}"
    )

    print(
        f"Processed X_test shape: "
        f"{X_test_processed.shape}"
    )

    return X_train_processed, X_test_processed, preprocessor


# =========================================================
# Main execution
# =========================================================

if __name__ == "__main__":

    # 1. Load dataset
    df = load_data()

    # 2. Remove unnecessary columns
    df = remove_unnecessary_columns(df)

    # 3. Check missing values
    check_missing_values(df)

    # 4. Split features and target
    X, y = split_data(df)

    # 5. Create train/test split
    X_train, X_test, y_train, y_test = create_train_test_split(
        X,
        y
    )

    # 6. Preprocess data
    X_train_processed, X_test_processed, preprocessor = preprocess_data(
        X_train,
        X_test
    )

    print("\n✓ Data preprocessing completed successfully")