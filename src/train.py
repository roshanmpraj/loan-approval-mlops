import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from data_cleaning import (
    load_data,
    remove_unnecessary_columns,
    split_data,
    create_train_test_split,
    preprocess_data,
)


# =========================================================
# Train Logistic Regression
# =========================================================

def train_logistic_regression(X_train, y_train):

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        random_state=42
    )

    model.fit(X_train, y_train)

    print("✓ Logistic Regression training completed")

    return model


# =========================================================
# Train Random Forest
# =========================================================

def train_random_forest(X_train, y_train):

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("✓ Random Forest training completed")

    return model


# =========================================================
# Main execution
# =========================================================

if __name__ == "__main__":

    # 1. Load dataset
    df = load_data()

    # 2. Remove unnecessary columns
    df = remove_unnecessary_columns(df)

    # 3. Split features and target
    X, y = split_data(df)

    # 4. Create train/test split
    X_train, X_test, y_train, y_test = create_train_test_split(
        X,
        y
    )

    # 5. Preprocess data
    X_train_processed, X_test_processed, preprocessor = preprocess_data(
        X_train,
        X_test
    )

    # 6. Train Logistic Regression
    logistic_model = train_logistic_regression(
        X_train_processed,
        y_train
    )

    # 7. Train Random Forest
    random_forest_model = train_random_forest(
        X_train_processed,
        y_train
    )

    print("\n===================================")
    print("MODEL TRAINING COMPLETED")
    print("===================================")
