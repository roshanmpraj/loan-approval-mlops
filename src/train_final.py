import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from data_cleaning import (
    load_data,
    remove_unnecessary_columns,
    split_data,
    create_train_test_split,
    create_preprocessing_pipeline,
)


# =========================================================
# Configuration
# =========================================================

MODEL_PATH = "models/loan_approval_model.pkl"

BEST_C = 1


# =========================================================
# Create Final Model Pipeline
# =========================================================

def create_final_pipeline():

    print("\nCreating final model pipeline...")

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline()

    # Create Logistic Regression model
    model = LogisticRegression(
        C=BEST_C,
        max_iter=2000,
        solver="liblinear",
        random_state=42
    )

    # Combine preprocessing + model
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    print("✓ Final pipeline created")

    return pipeline


# =========================================================
# Train Final Model
# =========================================================

def train_final_model(X_train, y_train):

    print("\nTraining final Logistic Regression model...")

    pipeline = create_final_pipeline()

    # Train preprocessing and model together
    pipeline.fit(
        X_train,
        y_train
    )

    print("✓ Final model training completed")

    return pipeline


# =========================================================
# Save Model
# =========================================================

def save_model(model, model_path):

    print("\nSaving final model...")

    # Make sure models directory exists
    os.makedirs(
        os.path.dirname(model_path),
        exist_ok=True
    )

    # Save complete pipeline
    joblib.dump(
        model,
        model_path
    )

    print(f"✓ Model saved to: {model_path}")


# =========================================================
# Main Execution
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # 1. Load dataset
    # -----------------------------------------------------

    df = load_data()

    # -----------------------------------------------------
    # 2. Remove unnecessary columns
    # -----------------------------------------------------

    df = remove_unnecessary_columns(df)

    # -----------------------------------------------------
    # 3. Split features and target
    # -----------------------------------------------------

    X, y = split_data(df)

    # -----------------------------------------------------
    # 4. Create train/test split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = create_train_test_split(
        X,
        y
    )

    # -----------------------------------------------------
    # 5. Train final model
    # -----------------------------------------------------

    final_model = train_final_model(
        X_train,
        y_train
    )

    # -----------------------------------------------------
    # 6. Save final model
    # -----------------------------------------------------

    save_model(
        final_model,
        MODEL_PATH
    )

    print("\n===================================")
    print("FINAL MODEL TRAINING COMPLETED")
    print("===================================")
