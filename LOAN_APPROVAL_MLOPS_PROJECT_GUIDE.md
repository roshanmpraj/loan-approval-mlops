# Loan Approval Classification — End-to-End MLOps Project Guide

> **Project status:** Completed and validated on 2026-09-07  
> **Repository:** `roshanmpraj/loan-approval-mlops`  
> **ML problem:** Binary classification — predict whether a loan application is approved.  
> **Deployment:** Docker + Amazon ECR + Amazon EKS + AWS Load Balancer  
> **CI/CD:** GitHub Actions  
> **API:** FastAPI  
> **Experiment tracking:** MLflow

---

## 1. What this project demonstrates

This project takes a traditional machine-learning classification problem and turns it into a production-style MLOps workflow.

```text
Developer
   |
   | git push
   v
GitHub
   |
   v
GitHub Actions
   |
   +--> Data tests
   +--> Data validation
   +--> Model training
   +--> Hyperparameter tuning
   +--> Model evaluation
   +--> MLflow tracking
   +--> Docker build
   +--> Push image to Amazon ECR
   +--> Deploy image to Amazon EKS
   |
   v
AWS EKS
   |
   v
Kubernetes Service (LoadBalancer)
   |
   v
FastAPI
   |
   v
Loan Approval Prediction
```

The final production test returned:

```json
{
  "prediction": "APPROVED",
  "approval_probability": 75.42
}
```

---

# 2. Technology stack

| Area | Technology |
|---|---|
| Language | Python |
| Data processing | pandas, NumPy |
| ML | scikit-learn |
| Models | Logistic Regression, Random Forest |
| Hyperparameter tuning | GridSearchCV |
| Experiment tracking | MLflow |
| API | FastAPI + Uvicorn |
| Testing | pytest |
| Containerization | Docker |
| Container registry | Amazon ECR |
| Orchestration | Kubernetes |
| Cloud Kubernetes | Amazon EKS |
| CI/CD | GitHub Actions |
| AWS authentication | GitHub OIDC |
| Cloud region | `ap-south-1` |

---

# 3. Project structure

Final project structure:

```text
loan-approval-mlops/
│
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml
│
├── data/
│   └── loan_approval.csv
│
├── k8s/
│   └── deployment.yaml
│
├── models/
│   └── loan_approval_model.pkl
│
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py
│   ├── data_validation.py
│   ├── evaluate.py
│   ├── evaluate_tuned.py
│   ├── inspect_data.py
│   ├── mlflow_tracking.py
│   ├── predict.py
│   ├── train.py
│   ├── train_final.py
│   ├── train_pipeline.py
│   └── tuning.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data_validation.py
│   └── test_prediction.py
│
├── app.py
├── Dockerfile
├── requirements.txt
└── README.md
```

Files such as `mlflow.db`, `mlruns/`, Python caches, and the generated `.pkl` model should normally be ignored from Git.

---

# 4. Phase 1 — Create the project

## Step 1 — Create a virtual environment

```bash
mkdir loan-approval-mlops
cd loan-approval-mlops

python3 -m venv .venv
source .venv/bin/activate
```

Verify:

```bash
python --version
pip --version
```

Why?

The virtual environment isolates project dependencies from other Python projects.

---

# 5. Phase 2 — Install dependencies

Create `requirements.txt`:

```text
numpy
pandas
scikit-learn
joblib
mlflow
pytest
fastapi
uvicorn
```

Install:

```bash
pip install -r requirements.txt
```

Verify:

```bash
pip list
```

---

# 6. Phase 3 — Dataset

The dataset is:

```text
data/loan_approval.csv
```

The dataset contains:

```text
Loan_ID
Gender
Married
Dependents
Education
Self_Employed
ApplicantIncome
CoapplicantIncome
LoanAmount
Loan_Amount_Term
Credit_History
Property_Area
Loan_Status
```

Target:

```text
Loan_Status
```

Allowed target values:

```text
Y = Approved
N = Not approved
```

---

# 7. Phase 4 — Data validation

Create `src/data_validation.py`.

```python
import pandas as pd

DATA_PATH = "data/loan_approval.csv"

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


def load_data():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    return df


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


def check_missing_values(df):
    print("\nChecking missing values...")

    missing = df.isnull().sum()

    print(missing)


def check_duplicates(df):
    print("\nChecking duplicates...")

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate rows: {duplicate_count}")


if __name__ == "__main__":

    df = load_data()

    validate_schema(df)

    validate_target(df)

    check_missing_values(df)

    check_duplicates(df)

    print("\n✓ Data validation completed successfully")
```

Run:

```bash
python src/data_validation.py
```

Expected:

```text
✓ Schema validation passed
✓ Target validation passed
✓ Data validation completed successfully
```

---

# 8. Phase 5 — Automated data tests

Create `tests/test_data_validation.py`.

```python
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

    assert set(
        df["Loan_Status"].dropna().unique()
    ) <= {"Y", "N"}
```

Run:

```bash
pytest -v tests/test_data_validation.py
```

Why?

These tests stop the pipeline early if the dataset structure changes unexpectedly.

---

# 9. Phase 6 — Data cleaning and preprocessing

Important production principle:

> Do preprocessing inside a scikit-learn Pipeline so training and inference use exactly the same transformations.

Numerical columns:

```python
NUMERICAL_COLUMNS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]
```

Categorical columns:

```python
CATEGORICAL_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]
```

Numerical preprocessing:

```python
Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)
```

Categorical preprocessing:

```python
Pipeline(
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
```

Combine them using:

```python
ColumnTransformer(
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
```

### Why?

This prevents training/inference preprocessing mismatch.

It also prevents leakage because the preprocessing parameters are learned from training data and then applied to validation/test data.

---

# 10. Train/test split

The project uses:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Meaning:

```text
80% → training
20% → test
```

`stratify=y` preserves the class distribution.

`random_state=42` makes the split reproducible.

---

# 11. Phase 7 — Model training and tuning

Two candidate models were used:

```text
1. Logistic Regression
2. Random Forest
```

The project uses F1 as the model-selection metric.

```python
F1_SCORER = make_scorer(
    f1_score,
    pos_label="Y"
)
```

Why F1?

Loan approval is a classification problem where both precision and recall matter.

F1 balances:

```text
Precision + Recall
```

---

# 12. Logistic Regression tuning

Pipeline:

```python
Pipeline(
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
```

Parameter grid:

```python
param_grid = {
    "model__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ]
}
```

Grid search:

```python
GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring=F1_SCORER,
    n_jobs=-1,
    verbose=1
)
```

Important:

```text
C
↓
inverse regularization strength
```

Smaller `C` → stronger regularization.

Larger `C` → weaker regularization.

---

# 13. Random Forest tuning

Model:

```python
RandomForestClassifier(
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
```

Parameter grid:

```python
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
```

Again:

```python
GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring=F1_SCORER,
    n_jobs=-1
)
```

The best CV F1 score from each model is compared.

---

# 14. Model selection

Conceptually:

```text
Logistic Regression
       |
       | 5-fold CV F1
       v
    Score A

Random Forest
       |
       | 5-fold CV F1
       v
    Score B

Compare A vs B
       |
       v
Select higher score
```

The project compares:

```python
if logistic_score >= random_forest_score:
    selected = logistic
else:
    selected = random_forest
```

---

# 15. Model evaluation

The selected model is evaluated using:

```text
Accuracy
Precision
Recall
F1
ROC-AUC
Confusion Matrix
```

Example:

```python
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
```

ROC-AUC:

```python
y_test_binary = y_test.map({
    "N": 0,
    "Y": 1
})

roc_auc = roc_auc_score(
    y_test_binary,
    y_probability
)
```

---

# 16. Phase 8 — MLflow

The training pipeline creates an MLflow experiment:

```python
mlflow.set_experiment(
    "loan-approval-classification"
)
```

During the final run, it logs:

```text
Model name
Hyperparameters
Accuracy
Precision
Recall
F1
ROC-AUC
Model artifact
```

Example:

```python
with mlflow.start_run(
    run_name=f"final-{best_model_name.lower().replace(' ', '-')}"
):

    mlflow.log_param(
        "model",
        best_model_name
    )

    for parameter, value in best_params.items():
        mlflow.log_param(
            parameter,
            str(value)
        )

    mlflow.log_metrics(
        test_metrics
    )

    mlflow.sklearn.log_model(
        best_model,
        name="model",
        skops_trusted_types=[
            "numpy.dtype"
        ]
    )
```

Why MLflow?

It provides experiment tracking and model artifact management instead of relying only on console output.

---

# 17. Save the final model

The complete preprocessing + model pipeline is saved as:

```text
models/loan_approval_model.pkl
```

Code:

```python
joblib.dump(
    best_model,
    "models/loan_approval_model.pkl"
)
```

Important:

The saved object is the complete pipeline, not only the estimator.

Therefore inference can receive raw application fields.

---

# 18. Run the complete ML pipeline

The main training command is:

```bash
python src/train_pipeline.py
```

This performs:

```text
Load data
   ↓
Remove Loan_ID
   ↓
Split X / y
   ↓
Train/test split
   ↓
Tune Logistic Regression
   ↓
Tune Random Forest
   ↓
Compare CV F1
   ↓
Select best model
   ↓
Evaluate on test data
   ↓
Log results to MLflow
   ↓
Save model
```

---

# 19. Phase 9 — Prediction test

Create `tests/test_prediction.py`.

```python
import joblib
import pandas as pd

MODEL_PATH = "models/loan_approval_model.pkl"


def test_model_prediction():

    model = joblib.load(MODEL_PATH)

    applicant = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 2000,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Property_Area": "Urban",
    }

    applicant_df = pd.DataFrame([applicant])

    prediction = model.predict(
        applicant_df
    )[0]

    assert prediction in ["Y", "N"]
```

Run:

```bash
pytest -v
```

Expected:

```text
5 passed
```

---

# 20. Phase 10 — FastAPI

`app.py` exposes the model through REST.

Core implementation:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Loan Approval Prediction API",
    version="1.0.0"
)

model = joblib.load(
    "models/loan_approval_model.pkl"
)


class LoanApplication(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "loan-approval-api"
    }


@app.post("/predict")
def predict(application: LoanApplication):

    data = pd.DataFrame([
        application.model_dump()
    ])

    prediction = model.predict(data)[0]

    probability = (
        model.predict_proba(data)[0][1] * 100
    )

    return {
        "prediction": (
            "APPROVED"
            if prediction == "Y"
            else "REJECTED"
        ),
        "approval_probability": round(
            float(probability),
            2
        )
    }
```

---

# 21. Run API locally

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/docs
```

FastAPI automatically provides Swagger/OpenAPI documentation.

---

# 22. Test the API

```bash
curl -X POST \
  http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 2000,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": "Urban"
  }'
```

---

# 23. Phase 11 — Docker

Dockerfile:

```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY models ./models
COPY src ./src

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build:

```bash
docker build -t loan-approval-api:local .
```

Run:

```bash
docker run -p 8000:8000 loan-approval-api:local
```

Test:

```bash
curl http://localhost:8000/
```

---

# 24. Apple Silicon / EKS architecture issue

If developing on an Apple Silicon Mac, Docker may build an ARM64 image.

EKS worker nodes may be x86_64.

That can cause:

```text
ErrImagePull
```

or architecture mismatch errors.

Build the deployment image explicitly for AMD64:

```bash
docker build \
  --platform linux/amd64 \
  -t loan-approval-api:amd64 .
```

For GitHub Actions, the workflow also uses:

```bash
docker build --platform linux/amd64 ...
```

This ensures compatibility with the EKS node architecture used in this project.

---

# 25. Phase 12 — Kubernetes

`k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: loan-approval-api
spec:
  replicas: 1

  selector:
    matchLabels:
      app: loan-approval-api

  template:
    metadata:
      labels:
        app: loan-approval-api

    spec:
      containers:
        - name: loan-approval-api

          image: <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/loan-approval-api:latest

          ports:
            - containerPort: 8000

---
apiVersion: v1
kind: Service
metadata:
  name: loan-approval-api

spec:
  selector:
    app: loan-approval-api

  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000

  type: LoadBalancer
```

Deploy:

```bash
kubectl apply -f k8s/deployment.yaml
```

Check:

```bash
kubectl get pods
```

Check service:

```bash
kubectl get service loan-approval-api
```

The service gets an AWS Load Balancer DNS name.

---

# 26. Phase 13 — AWS EKS

AWS region:

```bash
ap-south-1
```

Cluster:

```text
loan-approval-cluster
```

Node group:

```text
loan-workers
```

Node type used:

```text
t3.small
```

Cluster creation was done with `eksctl`.

Example pattern:

```bash
eksctl create cluster \
  --name loan-approval-cluster \
  --region ap-south-1 \
  --nodes 1 \
  --node-type t3.small
```

Then verify:

```bash
aws eks update-kubeconfig \
  --region ap-south-1 \
  --name loan-approval-cluster
```

```bash
kubectl get nodes
```

Expected:

```text
STATUS
Ready
```

---

# 27. Phase 14 — Amazon ECR

Create the repository:

```bash
aws ecr create-repository \
  --repository-name loan-approval-api \
  --region ap-south-1
```

Login:

```bash
aws ecr get-login-password \
  --region ap-south-1 |
docker login \
  --username AWS \
  --password-stdin \
  <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com
```

Build:

```bash
docker build \
  --platform linux/amd64 \
  -t loan-approval-api:amd64 .
```

Tag:

```bash
docker tag \
  loan-approval-api:amd64 \
  <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/loan-approval-api:latest
```

Push:

```bash
docker push \
  <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/loan-approval-api:latest
```

---

# 28. Phase 15 — GitHub Actions

Workflow:

`.github/workflows/ml-pipeline.yml`

```yaml
name: Loan Approval ML Pipeline

on:
  push:
    branches:
      - main

permissions:
  id-token: write
  contents: read

env:
  AWS_REGION: ap-south-1
  ECR_REGISTRY: <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com
  ECR_REPOSITORY: loan-approval-api

jobs:
  ml-pipeline:
    runs-on: ubuntu-latest

    steps:

      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run data tests
        run: pytest tests/test_data_validation.py

      - name: Validate data
        run: python src/data_validation.py

      - name: Train ML pipeline
        run: python src/train_pipeline.py

      - name: Run prediction test
        run: pytest tests/test_prediction.py

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v6
        with:
          role-to-assume: arn:aws:iam::<ACCOUNT_ID>:role/GitHubActionsLoanApprovalRole
          aws-region: ap-south-1

      - name: Login to Amazon ECR
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build Docker image
        run: |
          docker build --platform linux/amd64 \
            -t $ECR_REGISTRY/$ECR_REPOSITORY:${{ github.sha }} \
            -t $ECR_REGISTRY/$ECR_REPOSITORY:latest .

      - name: Push Docker image to ECR
        run: |
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:${{ github.sha }}
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest

      - name: Update kubeconfig
        run: |
          aws eks update-kubeconfig \
            --region $AWS_REGION \
            --name loan-approval-cluster

      - name: Deploy to EKS
        run: |
          kubectl set image deployment/loan-approval-api \
            loan-approval-api=$ECR_REGISTRY/$ECR_REPOSITORY:${{ github.sha }}

      - name: Wait for deployment
        run: |
          kubectl rollout status deployment/loan-approval-api --timeout=180s
```

---

# 29. Why GitHub OIDC?

Instead of storing a long-lived AWS access key in GitHub, GitHub Actions uses OIDC to assume an AWS IAM role.

Flow:

```text
GitHub Actions
      |
      | OIDC token
      v
AWS IAM
      |
      | AssumeRoleWithWebIdentity
      v
GitHubActionsLoanApprovalRole
      |
      +--> ECR permissions
      |
      +--> EKS permissions
```

The workflow uses:

```yaml
permissions:
  id-token: write
  contents: read
```

and:

```yaml
uses: aws-actions/configure-aws-credentials@v6
```

This is preferable to hard-coded AWS credentials.

---

# 30. EKS access for GitHub Actions

The GitHub Actions IAM role needs:

```text
eks:DescribeCluster
```

to obtain cluster information.

The role also needs Kubernetes authorization through an EKS Access Entry.

The project used:

```text
AmazonEKSEditPolicy
```

scoped to the `default` namespace.

This allows GitHub Actions to run:

```bash
kubectl set image ...
kubectl rollout status ...
```

---

# 31. CI/CD pipeline explained for an interview

When code is pushed to `main`:

```text
1. Checkout code
2. Install Python dependencies
3. Run data tests
4. Validate dataset
5. Train and tune models
6. Evaluate selected model
7. Track experiment in MLflow
8. Save model
9. Run prediction test
10. Authenticate to AWS through OIDC
11. Login to ECR
12. Build AMD64 Docker image
13. Tag image with Git SHA
14. Push image to ECR
15. Update EKS kubeconfig
16. Update Kubernetes deployment image
17. Wait for rollout
```

The Git SHA tag gives each deployment an immutable image version.

Example:

```text
loan-approval-api:<git-sha>
```

while:

```text
latest
```

points to the latest image.

---

# 32. Production validation

Check the pod:

```bash
kubectl get pods
```

Expected:

```text
1/1 Running
```

Check the service:

```bash
kubectl get service loan-approval-api
```

Expected:

```text
TYPE
LoadBalancer
```

Check FastAPI:

```bash
curl -I http://<LOAD_BALANCER_DNS>/docs
```

Expected:

```text
HTTP/1.1 200 OK
```

Check OpenAPI:

```bash
curl http://<LOAD_BALANCER_DNS>/openapi.json
```

The API exposes:

```text
GET  /
POST /predict
```

---

# 33. Production prediction test

```bash
curl -X POST \
  http://<LOAD_BALANCER_DNS>/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 2000,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": "Urban"
  }'
```

Validated result during this project:

```json
{
  "prediction": "APPROVED",
  "approval_probability": 75.42
}
```

---

# 34. Important interview explanation — why the saved model contains preprocessing

Without a pipeline:

```text
Training:
raw data
  ↓
manual preprocessing
  ↓
model
```

Inference could accidentally do:

```text
API input
  ↓
different preprocessing
  ↓
model
```

That creates training-serving skew.

With a scikit-learn Pipeline:

```text
raw input
   ↓
same preprocessing
   ↓
same trained model
   ↓
prediction
```

The entire pipeline is saved with `joblib`.

---

# 35. Important interview explanation — why use GridSearchCV?

Instead of manually choosing one value:

```text
C = 1
```

we test candidates:

```text
0.01
0.1
1
10
100
```

with 5-fold cross-validation.

For Random Forest:

```text
n_estimators
max_depth
min_samples_split
```

are searched in combinations.

The best combination according to F1 is selected.

---

# 36. Important interview explanation — why F1?

F1 is:

```text
2 × Precision × Recall
----------------------
   Precision + Recall
```

It is useful when we care about both false positives and false negatives.

For this project:

```text
Positive class = Y = Approved
```

Therefore the scorer explicitly uses:

```python
pos_label="Y"
```

---

# 37. Important interview explanation — Docker vs Kubernetes

Docker:

```text
Packages the application and dependencies.
```

Kubernetes:

```text
Runs and manages containers.
```

In this project:

```text
Docker
  ↓
Container image
  ↓
Amazon ECR
  ↓
Amazon EKS
```

---

# 38. Important interview explanation — ECR vs EKS

ECR:

```text
Container image registry
```

EKS:

```text
Managed Kubernetes control plane
```

So:

```text
ECR stores the image.
EKS runs the image.
```

---

# 39. Important interview explanation — LoadBalancer Service

Kubernetes Service:

```yaml
type: LoadBalancer
```

causes AWS to provision an external load balancer.

Request flow:

```text
Internet
   ↓
AWS Load Balancer
   ↓
Kubernetes Service
   ↓
Pod
   ↓
FastAPI
```

---

# 40. Important interview explanation — Git SHA deployment

The workflow builds:

```text
image:<github.sha>
```

and:

```text
image:latest
```

Deployment uses:

```text
image:<github.sha>
```

Why?

Because Git SHA identifies exactly which source revision produced the image.

This improves traceability and rollback capability.

---

# 41. What was actually implemented vs future improvements

Implemented:

```text
✓ Data validation
✓ Automated tests
✓ Preprocessing pipeline
✓ Logistic Regression
✓ Random Forest
✓ Hyperparameter tuning
✓ Cross-validation
✓ Model selection
✓ Model evaluation
✓ MLflow tracking
✓ Model artifact
✓ FastAPI
✓ Docker
✓ Amazon ECR
✓ Amazon EKS
✓ Kubernetes LoadBalancer
✓ GitHub Actions CI/CD
✓ GitHub OIDC
✓ Automated EKS deployment
✓ Production inference validation
```

Not implemented:

```text
✗ Prometheus/Grafana monitoring
✗ Centralized logging
✗ Model drift detection
✗ Feature store
✗ Automated rollback
✗ Blue/green deployment
✗ Canary deployment
✗ HTTPS/TLS configuration
✗ Production database
```

Do not claim these as implemented in an interview.

---

# 42. Final revision checklist

Before the interview, be able to explain:

## ML

- What is classification?
- Why Logistic Regression?
- Why Random Forest?
- What is regularization?
- What does `C` mean?
- What is cross-validation?
- What is GridSearchCV?
- Why F1?
- Precision vs recall?
- ROC-AUC?
- Confusion matrix?
- What is data leakage?
- Why use Pipeline?

## MLOps

- Why data validation?
- Why automated tests?
- Why MLflow?
- Why save the entire pipeline?
- What happens after `git push`?
- What does GitHub Actions do?
- Why Docker?
- Why ECR?
- Why EKS?
- Why Kubernetes?
- Why LoadBalancer?
- Why GitHub OIDC?

## AWS

- IAM
- OIDC
- ECR
- EKS
- IAM role
- EKS Access Entry
- Kubernetes Service
- Kubernetes Deployment

---

# 43. One-minute interview answer

> I built an end-to-end MLOps pipeline for a loan approval classification problem. The pipeline starts with schema and target validation and automated pytest checks. I used a scikit-learn preprocessing pipeline with median imputation for numerical features and most-frequent imputation plus one-hot encoding for categorical features.
>
> I trained and tuned Logistic Regression and Random Forest using GridSearchCV with 5-fold cross-validation and F1 as the selection metric. The best model was evaluated using accuracy, precision, recall, F1 and ROC-AUC, and the experiment, parameters, metrics and model were tracked with MLflow.
>
> The complete preprocessing and model pipeline was saved as a model artifact and exposed through FastAPI. I containerized the API with Docker and pushed the image to Amazon ECR.
>
> For CI/CD, I used GitHub Actions. A push to main triggers tests, data validation, model training, prediction testing, Docker build and ECR push. GitHub authenticates to AWS using OIDC instead of static AWS credentials. The workflow then updates the EKS deployment with the Git SHA image and waits for the Kubernetes rollout.
>
> The API runs in EKS behind an AWS Load Balancer. I validated the live endpoint with a real loan application and received an APPROVED prediction with a 75.42% approval probability.

---

# 44. Project completion record

Before deleting the AWS environment, record:

```text
Project: Loan Approval Classification MLOps

Repository:
roshanmpraj/loan-approval-mlops

Completed:
2026-09-07

Cloud:
AWS

Region:
ap-south-1

EKS:
loan-approval-cluster

ECR:
loan-approval-api

CI/CD:
GitHub Actions

Authentication:
GitHub OIDC → AWS IAM

API:
FastAPI

Production validation:
HTTP 200 /docs
POST /predict successful

Validated prediction:
APPROVED
Approval probability:
75.42%
```

After this record is committed to GitHub, the AWS environment can be safely removed.

---

# 45. AWS CLEANUP — IMPORTANT

Do this only after the project documentation and completion record have been pushed to GitHub.

## 45.1 Delete Kubernetes application

```bash
kubectl delete -f k8s/deployment.yaml
```

This removes the Kubernetes Deployment and LoadBalancer Service.

Verify:

```bash
kubectl get pods
kubectl get service
```

---

## 45.2 Delete EKS cluster

```bash
eksctl delete cluster \
  --name loan-approval-cluster \
  --region ap-south-1
```

Wait for deletion to complete.

Verify:

```bash
aws eks list-clusters \
  --region ap-south-1
```

---

## 45.3 Delete ECR repository

```bash
aws ecr delete-repository \
  --repository-name loan-approval-api \
  --region ap-south-1 \
  --force
```

Verify:

```bash
aws ecr describe-repositories \
  --repository-names loan-approval-api \
  --region ap-south-1
```

A `RepositoryNotFoundException` means it was deleted.

---

# 46. Delete the GitHub OIDC IAM access

First remove the inline policies:

```bash
aws iam delete-role-policy \
  --role-name GitHubActionsLoanApprovalRole \
  --policy-name LoanApprovalECRPushPolicy
```

```bash
aws iam delete-role-policy \
  --role-name GitHubActionsLoanApprovalRole \
  --policy-name LoanApprovalEKSDeployPolicy
```

Delete the role:

```bash
aws iam delete-role \
  --role-name GitHubActionsLoanApprovalRole
```

Delete the GitHub OIDC provider:

```bash
aws iam delete-open-id-connect-provider \
  --openid-connect-provider-arn \
  arn:aws:iam::<ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com
```

Verify:

```bash
aws iam list-open-id-connect-providers
```

---

# 47. Final AWS cost verification

Check EKS:

```bash
aws eks list-clusters \
  --region ap-south-1
```

Check ECR:

```bash
aws ecr describe-repositories \
  --region ap-south-1
```

Check EC2 instances:

```bash
aws ec2 describe-instances \
  --region ap-south-1 \
  --filters Name=instance-state-name,Values=running
```

Check load balancers if needed:

```bash
aws elbv2 describe-load-balancers \
  --region ap-south-1
```

The goal is to confirm that the EKS cluster, worker node and LoadBalancer created for this project no longer exist.

---

# 48. Final project story

The complete project demonstrates:

```text
Machine Learning
       +
Software Engineering
       +
Testing
       +
Experiment Tracking
       +
Docker
       +
CI/CD
       +
IAM/OIDC
       +
Container Registry
       +
Kubernetes
       +
AWS EKS
       =
End-to-End MLOps
```

The key interview message is:

> The project is not only a machine-learning model. It demonstrates how a model moves from source code and data validation through automated training, testing, containerization, registry management, cloud deployment, Kubernetes orchestration and production inference.
