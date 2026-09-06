from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer

from data_cleaning import (
    load_data,
    remove_unnecessary_columns,
    split_data,
    create_train_test_split,
    create_preprocessing_pipeline,
)


# =========================================================
# F1 Scorer
# =========================================================

# Y = Approved
# N = Not Approved
#
# We explicitly tell sklearn that Y is the positive class.

F1_SCORER = make_scorer(
    f1_score,
    pos_label="Y"
)


# =========================================================
# Create Logistic Regression Pipeline
# =========================================================

def create_logistic_pipeline():

    preprocessor = create_preprocessing_pipeline()

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    random_state=42
                )
            )
        ]
    )

    return pipeline


# =========================================================
# Create Random Forest Pipeline
# =========================================================

def create_random_forest_pipeline():

    preprocessor = create_preprocessing_pipeline()

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1
                )
            )
        ]
    )

    return pipeline


# =========================================================
# Tune Logistic Regression
# =========================================================

def tune_logistic_regression(X_train, y_train):

    print("\n===================================")
    print("TUNING LOGISTIC REGRESSION")
    print("===================================")

    pipeline = create_logistic_pipeline()

    param_grid = {
        "model__C": [
            0.01,
            0.1,
            1,
            10,
            100
        ]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring=F1_SCORER,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print(
        f"Best CV F1 Score: "
        f"{grid_search.best_score_:.4f}"
    )

    return grid_search


# =========================================================
# Tune Random Forest
# =========================================================

def tune_random_forest(X_train, y_train):

    print("\n===================================")
    print("TUNING RANDOM FOREST")
    print("===================================")

    pipeline = create_random_forest_pipeline()

    param_grid = {
        "model__n_estimators": [
            100,
            200,
            300
        ],
        "model__max_depth": [
            5,
            10,
            None
        ],
        "model__min_samples_split": [
            2,
            5,
            10
        ]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring=F1_SCORER,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print(
        f"Best CV F1 Score: "
        f"{grid_search.best_score_:.4f}"
    )

    return grid_search


# =========================================================
# Main Execution
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

    # 5. Tune Logistic Regression
    logistic_search = tune_logistic_regression(
        X_train,
        y_train
    )

    # 6. Tune Random Forest
    random_forest_search = tune_random_forest(
        X_train,
        y_train
    )

    # 7. Tuning completed
    print("\n===================================")
    print("HYPERPARAMETER TUNING COMPLETED")
    print("===================================")