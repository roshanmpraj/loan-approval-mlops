import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from data_cleaning import (
    load_data,
    remove_unnecessary_columns,
    split_data,
    create_train_test_split,
)

from tuning import (
    tune_logistic_regression,
    tune_random_forest,
)


# =========================================================
# Configuration
# =========================================================

EXPERIMENT_NAME = "loan-approval-classification"

MODEL_PATH = "models/loan_approval_model.pkl"


# =========================================================
# Evaluate Model
# =========================================================

def evaluate_model(model, X_test, y_test):

    print("\nEvaluating selected model...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Get probability for positive class
    # Y = Approved
    y_probability = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label="Y"
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label="Y"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="Y"
    )

    # Convert target to binary for ROC-AUC
    y_test_binary = y_test.map({
        "N": 0,
        "Y": 1
    })

    roc_auc = roc_auc_score(
        y_test_binary,
        y_probability
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["N", "Y"]
    )

    print("\nTest Set Results")
    print("-----------------------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix")
    print("-----------------------------------")
    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }


# =========================================================
# Select Best Model
# =========================================================

def select_best_model(
    logistic_search,
    random_forest_search
):

    logistic_score = (
        logistic_search.best_score_
    )

    random_forest_score = (
        random_forest_search.best_score_
    )

    print("\n===================================")
    print("MODEL SELECTION")
    print("===================================")

    print(
        f"Logistic Regression CV F1: "
        f"{logistic_score:.4f}"
    )

    print(
        f"Random Forest CV F1      : "
        f"{random_forest_score:.4f}"
    )

    if logistic_score >= random_forest_score:

        print("\n✓ Selected: Logistic Regression")

        return (
            "Logistic Regression",
            logistic_search.best_estimator_,
            logistic_search.best_params_,
        )

    else:

        print("\n✓ Selected: Random Forest")

        return (
            "Random Forest",
            random_forest_search.best_estimator_,
            random_forest_search.best_params_,
        )


# =========================================================
# Save Final Model
# =========================================================

def save_model(model):

    print("\nSaving final model...")

    # Make sure models directory exists
    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )

    # Save complete preprocessing + model pipeline
    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"✓ Model saved to: {MODEL_PATH}"
    )


# =========================================================
# Main Pipeline
# =========================================================

if __name__ == "__main__":

    print("\n===================================")
    print("LOAN APPROVAL ML PIPELINE")
    print("===================================")

    # -----------------------------------------------------
    # 1. Set MLflow experiment
    # -----------------------------------------------------

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    # -----------------------------------------------------
    # 2. Load dataset
    # -----------------------------------------------------

    df = load_data()

    # -----------------------------------------------------
    # 3. Remove unnecessary columns
    # -----------------------------------------------------

    df = remove_unnecessary_columns(df)

    # -----------------------------------------------------
    # 4. Split features and target
    # -----------------------------------------------------

    X, y = split_data(df)

    # -----------------------------------------------------
    # 5. Create train/test split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = (
        create_train_test_split(
            X,
            y
        )
    )

    # -----------------------------------------------------
    # 6. Tune Logistic Regression
    # -----------------------------------------------------

    logistic_search = (
        tune_logistic_regression(
            X_train,
            y_train
        )
    )

    # -----------------------------------------------------
    # 7. Tune Random Forest
    # -----------------------------------------------------

    random_forest_search = (
        tune_random_forest(
            X_train,
            y_train
        )
    )

    # -----------------------------------------------------
    # 8. Select best model
    # -----------------------------------------------------

    (
        best_model_name,
        best_model,
        best_params,
    ) = select_best_model(
        logistic_search,
        random_forest_search
    )

    # -----------------------------------------------------
    # 9. Evaluate selected model
    # -----------------------------------------------------

    test_metrics = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    # -----------------------------------------------------
    # 10. Start MLflow run
    # -----------------------------------------------------

    with mlflow.start_run(
        run_name=f"final-{best_model_name.lower().replace(' ', '-')}"
    ):

        # -------------------------------------------------
        # Log model name
        # -------------------------------------------------

        mlflow.log_param(
            "model",
            best_model_name
        )

        # -------------------------------------------------
        # Log best hyperparameters
        # -------------------------------------------------

        for parameter, value in best_params.items():

            # Convert values to strings where necessary
            # so MLflow can safely log them.

            mlflow.log_param(
                parameter,
                str(value)
            )

        # -------------------------------------------------
        # Log test metrics
        # -------------------------------------------------

        mlflow.log_metrics(
            test_metrics
        )

        # -------------------------------------------------
        # Log final model
        # -------------------------------------------------

        mlflow.sklearn.log_model(
            best_model,
            name="model",
            skops_trusted_types=[
                "numpy.dtype"
            ]
        )

        print("\n✓ Results logged to MLflow")

    # -----------------------------------------------------
    # 11. Save final model locally
    # -----------------------------------------------------

    save_model(
        best_model
    )

    print("\n===================================")
    print("ML PIPELINE COMPLETED")
    print("===================================")
