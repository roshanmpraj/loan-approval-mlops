import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from data_cleaning import (
    load_data,
    remove_unnecessary_columns,
    split_data,
    create_train_test_split,
    create_preprocessing_pipeline,
)


# =========================================================
# MLflow Configuration
# =========================================================

EXPERIMENT_NAME = "loan-approval-classification"


# =========================================================
# Create Model Pipeline
# =========================================================

def create_logistic_pipeline():

    preprocessor = create_preprocessing_pipeline()

    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LogisticRegression(
                    C=1,
                    max_iter=2000,
                    solver="liblinear",
                    random_state=42
                )
            )
        ]
    )

    return pipeline


def create_random_forest_pipeline():

    preprocessor = create_preprocessing_pipeline()

    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=5,
                    min_samples_split=5,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1
                )
            )
        ]
    )

    return pipeline


# =========================================================
# Evaluate Model
# =========================================================

def calculate_metrics(model, X_test, y_test):

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(X_test)[:, 1]

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

    y_test_binary = y_test.map({
        "N": 0,
        "Y": 1
    })

    roc_auc = roc_auc_score(
        y_test_binary,
        y_probability
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }


# =========================================================
# Track Logistic Regression
# =========================================================

def track_logistic_regression(
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n===================================")
    print("MLFLOW - LOGISTIC REGRESSION")
    print("===================================")

    model = create_logistic_pipeline()

    with mlflow.start_run(
        run_name="logistic-regression"
    ):

        # Train model
        model.fit(
            X_train,
            y_train
        )

        # Model parameters
        mlflow.log_param(
            "model",
            "LogisticRegression"
        )

        mlflow.log_param(
            "C",
            1
        )

        mlflow.log_param(
            "max_iter",
            2000
        )

        mlflow.log_param(
            "solver",
            "liblinear"
        )

        # Calculate metrics
        metrics = calculate_metrics(
            model,
            X_test,
            y_test
        )

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            skops_trusted_types=["numpy.dtype"]
        )

        print("\nMetrics:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

    return model


# =========================================================
# Track Random Forest
# =========================================================

def track_random_forest(
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n===================================")
    print("MLFLOW - RANDOM FOREST")
    print("===================================")

    model = create_random_forest_pipeline()

    with mlflow.start_run(
        run_name="random-forest"
    ):

        # Train model
        model.fit(
            X_train,
            y_train
        )

        # Model parameters
        mlflow.log_param(
            "model",
            "RandomForestClassifier"
        )

        mlflow.log_param(
            "n_estimators",
            200
        )

        mlflow.log_param(
            "max_depth",
            5
        )

        mlflow.log_param(
            "min_samples_split",
            5
        )

        mlflow.log_param(
            "class_weight",
            "balanced"
        )

        # Calculate metrics
        metrics = calculate_metrics(
            model,
            X_test,
            y_test
        )

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            skops_trusted_types=["numpy.dtype"]
        )

        print("\nMetrics:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

    return model


# =========================================================
# Main Execution
# =========================================================

if __name__ == "__main__":

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

    X_train, X_test, y_train, y_test = create_train_test_split(
        X,
        y
    )

    # -----------------------------------------------------
    # 6. Track Logistic Regression
    # -----------------------------------------------------

    logistic_model = track_logistic_regression(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # -----------------------------------------------------
    # 7. Track Random Forest
    # -----------------------------------------------------

    random_forest_model = track_random_forest(
        X_train,
        y_train,
        X_test,
        y_test
    )

    print("\n===================================")
    print("MLFLOW TRACKING COMPLETED")
    print("===================================")
