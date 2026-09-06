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
# Evaluate Model
# =========================================================

def evaluate_model(model, X_test, y_test, model_name):

    print(f"\nEvaluating {model_name}...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Get probability of the positive class
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

    # Convert:
    # N -> 0
    # Y -> 1
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

    # Print results
    print("\n-----------------------------------")
    print(f"{model_name} - TEST SET RESULTS")
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
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }


# =========================================================
# Compare Models
# =========================================================

def compare_models(results):

    print("\n===================================")
    print("TUNED MODEL COMPARISON")
    print("===================================")

    for result in results:

        print(f"\n{result['model']}")

        print(f"Accuracy : {result['accuracy']:.4f}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall   : {result['recall']:.4f}")
        print(f"F1 Score : {result['f1_score']:.4f}")
        print(f"ROC-AUC  : {result['roc_auc']:.4f}")


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
    # 5. Tune Logistic Regression
    # -----------------------------------------------------

    logistic_search = tune_logistic_regression(
        X_train,
        y_train
    )

    # -----------------------------------------------------
    # 6. Tune Random Forest
    # -----------------------------------------------------

    random_forest_search = tune_random_forest(
        X_train,
        y_train
    )

    # -----------------------------------------------------
    # 7. Get best tuned models
    # -----------------------------------------------------

    best_logistic_model = (
        logistic_search.best_estimator_
    )

    best_random_forest_model = (
        random_forest_search.best_estimator_
    )

    # -----------------------------------------------------
    # 8. Evaluate Logistic Regression
    # -----------------------------------------------------

    logistic_results = evaluate_model(
        best_logistic_model,
        X_test,
        y_test,
        "Tuned Logistic Regression"
    )

    # -----------------------------------------------------
    # 9. Evaluate Random Forest
    # -----------------------------------------------------

    random_forest_results = evaluate_model(
        best_random_forest_model,
        X_test,
        y_test,
        "Tuned Random Forest"
    )

    # -----------------------------------------------------
    # 10. Compare models
    # -----------------------------------------------------

    compare_models([
        logistic_results,
        random_forest_results
    ])

    print(
        "\n✓ Tuned model evaluation completed successfully"
    )
