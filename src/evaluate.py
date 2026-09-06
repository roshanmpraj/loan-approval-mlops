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
    preprocess_data,
)

from train import (
    train_logistic_regression,
    train_random_forest,
)


# =========================================================
# Evaluate Model
# =========================================================

def evaluate_model(model, X_test, y_test, model_name):

    print(f"\nEvaluating {model_name}...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate probability for ROC-AUC
    y_probability = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

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

    roc_auc = roc_auc_score(
        y_test.map({"N": 0, "Y": 1}),
        y_probability
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["N", "Y"]
    )

    print(f"\n{model_name} Results")
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
    print("MODEL COMPARISON")
    print("===================================")

    for result in results:

        print(f"\n{result['model']}")

        print(f"Accuracy : {result['accuracy']:.4f}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall   : {result['recall']:.4f}")
        print(f"F1 Score : {result['f1_score']:.4f}")
        print(f"ROC-AUC  : {result['roc_auc']:.4f}")


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

    # 8. Evaluate Logistic Regression
    logistic_results = evaluate_model(
        logistic_model,
        X_test_processed,
        y_test,
        "Logistic Regression"
    )

    # 9. Evaluate Random Forest
    random_forest_results = evaluate_model(
        random_forest_model,
        X_test_processed,
        y_test,
        "Random Forest"
    )

    # 10. Compare models
    compare_models([
        logistic_results,
        random_forest_results
    ])

    print("\n✓ Model evaluation completed successfully")
