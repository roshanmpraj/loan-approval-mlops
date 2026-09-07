# Loan Approval MLOps Project — End-to-End Guide

This document covers the complete implementation flow from local development through successful AWS EKS deployment using GitHub Actions.

> Scope: implementation and deployment only. Interview preparation and AWS deletion/cleanup steps are intentionally excluded.

---

## 1. Project Objective

Build a production-style machine learning application that:

1. Validates loan application data.
2. Cleans and preprocesses the data.
3. Trains multiple classification models.
4. Tunes hyperparameters using cross-validation.
5. Selects the best model.
6. Evaluates the model.
7. Tracks experiments with MLflow.
8. Saves the trained model.
9. Exposes predictions through FastAPI.
10. Packages the application using Docker.
11. Pushes the Docker image to Amazon ECR.
12. Deploys the application to Amazon EKS.
13. Automates the ML and deployment workflow using GitHub Actions.

---

## 2. Architecture

```text
Developer
   |
   v
GitHub Repository
   |
   v
GitHub Actions
   |
   +--> Data Tests
   |
   +--> Data Validation
   |
   +--> ML Training
   |
   +--> Prediction Test
   |
   +--> Docker Build
   |
   v
Amazon ECR
   |
   v
Amazon EKS
   |
   v
Kubernetes Service
   |
   v
AWS Load Balancer
   |
   v
FastAPI
   |
   v
/predict
   |
   v
Loan Approval Prediction
```

---

## 3. Technology Stack

| Area | Technology |
|---|---|
| Language | Python 3.11 |
| Data processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model serialization | Joblib |
| Experiment tracking | MLflow |
| API | FastAPI |
| API server | Uvicorn |
| Testing | Pytest |
| Containerization | Docker |
| Container registry | Amazon ECR |
| Orchestration | Kubernetes |
| Cloud Kubernetes | Amazon EKS |
| CI/CD | GitHub Actions |
| AWS authentication | GitHub OIDC + IAM |
| AWS region | ap-south-1 |

---

# 4. Repository Structure

```text
loan-approval-mlops/
|
+-- .github/
|   +-- workflows/
|       +-- ml-pipeline.yml
|
+-- data/
|   +-- loan_approval.csv
|
+-- models/
|   +-- loan_approval_model.pkl
|
+-- src/
|   +-- data_cleaning.py
|   +-- data_validation.py
|   +-- evaluate.py
|   +-- evaluate_tuned.py
|   +-- inspect_data.py
|   +-- mlflow_tracking.py
|   +-- predict.py
|   +-- train.py
|   +-- train_final.py
|   +-- train_pipeline.py
|   +-- tuning.py
|
+-- tests/
|   +-- test_data_validation.py
|   +-- test_prediction.py
|
+-- k8s/
|   +-- deployment.yaml
|
+-- app.py
+-- Dockerfile
+-- requirements.txt
+-- .gitignore
+-- README.md
```

---

# 5. Clone the Repository

### What are we doing?

Getting the project code locally.

### Commands

```bash
git clone https://github.com/roshanmpraj/loan-approval-mlops.git
cd loan-approval-mlops
```

### Verify

```bash
git status
```

Expected:

```text
On branch main
```

---

# 6. Create Python Virtual Environment

### What are we doing?

Creating an isolated Python environment for the project.

### Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Verify

```bash
python --version
```

---

# 7. Install Dependencies

### What are we doing?

Installing the libraries required by the ML pipeline and API.

### Command

```bash
pip install -r requirements.txt
```

Important dependencies include:

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

---

# 8. Dataset

Dataset:

```text
data/loan_approval.csv
```

Target:

```text
Loan_Status
```

Target values:

```text
Y = Approved
N = Rejected
```

Dataset shape:

```text
614 rows
13 columns
```

---

# 9. Data Validation

### What are we doing?

Checking the dataset before it enters the ML pipeline.

### File

```text
src/data_validation.py
```

### Expected columns

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

### Run

```bash
python src/data_validation.py
```

The validation checks:

- Dataset can be loaded.
- Expected columns exist.
- Target column exists.
- Target values are only `Y` and `N`.
- Missing values are reported.
- Duplicate rows are reported.

Expected final message:

```text
✓ Data validation completed successfully
```

---

# 10. Automated Data Tests

### What are we doing?

Making data validation repeatable and allowing invalid data to stop the pipeline.

### File

```text
tests/test_data_validation.py
```

Tests verify:

- Dataset exists.
- Dataset is not empty.
- Expected columns exist.
- Target column exists.
- Target values are valid.

### Run

```bash
pytest tests/test_data_validation.py -v
```

Expected:

```text
4 passed
```

---

# 11. Data Cleaning and Preprocessing

### What are we doing?

Converting raw data into a form suitable for machine learning.

### Numerical columns

```text
ApplicantIncome
CoapplicantIncome
LoanAmount
Loan_Amount_Term
Credit_History
```

Missing numerical values:

```text
Median imputation
```

### Categorical columns

```text
Gender
Married
Dependents
Education
Self_Employed
Property_Area
```

Missing categorical values:

```text
Most-frequent imputation
```

Categorical encoding:

```text
OneHotEncoder(handle_unknown="ignore")
```

---

# 12. Remove Identifier Column

`Loan_ID` is an identifier and is not used as a predictive feature.

It is removed before training:

```python
df = df.drop(columns=["Loan_ID"])
```

---

# 13. Train/Test Split

The data is split into:

```text
80% training
20% testing
```

Using:

```python
random_state=42
```

and:

```python
stratify=y
```

The preprocessing transformer is fitted on training data and then applied to test data.

This helps prevent data leakage.

---

# 14. Machine Learning Models

Two classification models are compared.

## Logistic Regression

```python
LogisticRegression(
    max_iter=2000,
    solver="liblinear",
    random_state=42
)
```

## Random Forest

```python
RandomForestClassifier(
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
```

---

# 15. Hyperparameter Tuning

### What are we doing?

Searching for better model configurations instead of relying on arbitrary parameter values.

The project uses:

```text
GridSearchCV
```

with:

```text
5-fold cross-validation
```

## Logistic Regression

Search:

```text
C = [0.01, 0.1, 1, 10, 100]
```

Total:

```text
5 combinations × 5 folds = 25 fits
```

## Random Forest

Search:

```text
n_estimators = [100, 200, 300]
max_depth = [5, 10, None]
min_samples_split = [2, 5, 10]
```

Total:

```text
3 × 3 × 3 = 27 combinations
27 × 5 folds = 135 fits
```

The tuning metric is:

```text
F1 score
```

with `Y` as the positive class.

---

# 16. Train the ML Pipeline

Main entry point:

```text
src/train_pipeline.py
```

### Run

```bash
python src/train_pipeline.py
```

### Pipeline

```text
Load data
   |
   v
Remove Loan_ID
   |
   v
Split X and y
   |
   v
Train/Test Split
   |
   v
Preprocessing
   |
   v
Tune Logistic Regression
   |
   v
Tune Random Forest
   |
   v
Select Best Model
   |
   v
Evaluate
   |
   v
MLflow Tracking
   |
   v
Save Model
```

The completed project selected:

```text
Logistic Regression
```

---

# 17. Model Evaluation

The model is evaluated using:

```text
Accuracy
Precision
Recall
F1 Score
ROC-AUC
Confusion Matrix
```

`Y` is treated as the positive class.

---

# 18. MLflow Tracking

### What are we doing?

Recording training information so experiments can be tracked.

Experiment:

```text
loan-approval-classification
```

Tracked information includes:

```text
Model name
Hyperparameters
Accuracy
Precision
Recall
F1
ROC-AUC
Confusion matrix
Model artifact
```

Local MLflow runtime files such as:

```text
mlruns/
mlflow.db
```

are excluded from Git.

---

# 19. Save the Model

The trained model is saved as:

```text
models/loan_approval_model.pkl
```

using Joblib.

### Verify

```bash
ls models/
```

Expected:

```text
loan_approval_model.pkl
```

---

# 20. Prediction Test

### What are we doing?

Checking that the saved model can be loaded and can produce a valid prediction.

### Run

```bash
pytest tests/test_prediction.py -v
```

Expected:

```text
1 passed
```

---

# 21. Run All Local Tests

Before Docker and AWS deployment:

```bash
pytest -v
```

Expected:

```text
5 passed
```

This is the local ML pipeline checkpoint.

---

# 22. FastAPI Application

### What are we doing?

Exposing the trained model through a REST API.

File:

```text
app.py
```

The application loads:

```text
models/loan_approval_model.pkl
```

---

# 23. API Endpoints

## Health check

```text
GET /
```

Response:

```json
{
  "status": "healthy",
  "service": "loan-approval-api"
}
```

## Prediction

```text
POST /predict
```

Input fields:

```text
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
```

---

# 24. Run FastAPI Locally

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Swagger UI can be used to test the API.

### Health check

```bash
curl http://127.0.0.1:8000/
```

Expected:

```json
{
  "status": "healthy",
  "service": "loan-approval-api"
}
```

---

# 25. Test Prediction Locally

```bash
curl -X POST http://127.0.0.1:8000/predict   -H "Content-Type: application/json"   -d '{
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
    "Property_Area": "Urban"
  }'
```

Example response:

```json
{
  "prediction": "APPROVED",
  "approval_probability": 78.47
}
```

---

# 26. Dockerize the Application

### What are we doing?

Packaging the API, model and dependencies into a portable container.

The Docker image contains:

```text
Python runtime
Dependencies
app.py
models/
src/
```

---

# 27. Dockerfile

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

---

# 28. Build Docker Image

```bash
docker build -t loan-approval-api:local .
```

Verify:

```bash
docker images
```

Expected image:

```text
loan-approval-api
```

---

# 29. Run Docker Locally

```bash
docker run -p 8000:8000 loan-approval-api:local
```

Test:

```bash
curl http://localhost:8000/
```

Then open:

```text
http://localhost:8000/docs
```

---

# 30. Apple Silicon / AMD64 Compatibility

When developing on Apple Silicon, Docker may build an ARM64 image.

The EKS worker environment used by this project is AMD64/x86_64.

Therefore the production image is explicitly built as:

```bash
docker build --platform linux/amd64 ...
```

Without this, Kubernetes can encounter errors such as:

```text
ErrImagePull
```

or architecture incompatibility.

---

# 31. Create Amazon ECR Repository

### What are we doing?

Creating a private registry to store the production Docker image.

AWS region:

```text
ap-south-1
```

Repository:

```text
loan-approval-api
```

### Command

```bash
aws ecr create-repository   --repository-name loan-approval-api   --region ap-south-1
```

### Verify

```bash
aws ecr describe-repositories   --repository-names loan-approval-api   --region ap-south-1
```

---

# 32. Login to ECR

```bash
aws ecr get-login-password   --region ap-south-1 | docker login   --username AWS   --password-stdin   517811334575.dkr.ecr.ap-south-1.amazonaws.com
```

Expected:

```text
Login Succeeded
```

---

# 33. Build Production Image

Because EKS uses AMD64:

```bash
docker build   --platform linux/amd64   -t loan-approval-api:amd64 .
```

---

# 34. Tag the Image

```bash
docker tag loan-approval-api:amd64   517811334575.dkr.ecr.ap-south-1.amazonaws.com/loan-approval-api:latest
```

---

# 35. Push Image to ECR

```bash
docker push   517811334575.dkr.ecr.ap-south-1.amazonaws.com/loan-approval-api:latest
```

Verify:

```bash
aws ecr describe-images   --repository-name loan-approval-api   --region ap-south-1
```

---

# 36. Create EKS Cluster

### What are we doing?

Creating the Kubernetes environment where the container will run.

Project configuration:

```text
Cluster: loan-approval-cluster
Region: ap-south-1
Node group: loan-workers
Instance: t3.small
Nodes: 1
```

### Command

```bash
eksctl create cluster   --name loan-approval-cluster   --region ap-south-1   --nodegroup-name loan-workers   --node-type t3.small   --nodes 1   --nodes-min 1   --nodes-max 1
```

---

# 37. Verify EKS

```bash
aws eks list-clusters --region ap-south-1
```

Expected:

```text
loan-approval-cluster
```

Configure kubectl:

```bash
aws eks update-kubeconfig   --region ap-south-1   --name loan-approval-cluster
```

Verify nodes:

```bash
kubectl get nodes
```

Expected:

```text
Ready
```

---

# 38. Kubernetes Manifest

File:

```text
k8s/deployment.yaml
```

It contains:

```text
Deployment
Service
```

The Deployment runs the application container.

The Service exposes the application.

The Service type is:

```text
LoadBalancer
```

---

# 39. Kubernetes Deployment

```bash
kubectl apply -f k8s/deployment.yaml
```

Expected:

```text
deployment.apps/loan-approval-api created
service/loan-approval-api created
```

---

# 40. Verify Pod

```bash
kubectl get pods
```

Expected:

```text
loan-approval-api-xxxxx   1/1   Running
```

Logs:

```bash
kubectl logs <pod-name>
```

Troubleshooting:

```bash
kubectl describe pod <pod-name>
```

---

# 41. Verify Kubernetes Service

```bash
kubectl get svc
```

Expected:

```text
loan-approval-api   LoadBalancer
```

Wait until an external DNS name is assigned.

Example:

```text
a113cf3cb16b646de9d6eae10432e2a1-604587778.ap-south-1.elb.amazonaws.com
```

---

# 42. Test the EKS Health Endpoint

Replace `<LOAD_BALANCER_DNS>` with the external DNS name:

```bash
curl http://<LOAD_BALANCER_DNS>/
```

Expected:

```json
{
  "status": "healthy",
  "service": "loan-approval-api"
}
```

Swagger:

```text
http://<LOAD_BALANCER_DNS>/docs
```

---

# 43. Test Production Prediction

```bash
curl -X POST http://<LOAD_BALANCER_DNS>/predict   -H "Content-Type: application/json"   -d '{
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
    "Property_Area": "Urban"
  }'
```

Successful production response:

```json
{
  "prediction": "APPROVED",
  "approval_probability": 78.47
}
```

At this point the application is running in EKS and serving predictions through the AWS Load Balancer.

---

# 44. GitHub Actions CI/CD

### What are we doing?

Automating the complete ML and deployment workflow.

Workflow file:

```text
.github/workflows/ml-pipeline.yml
```

Trigger:

```yaml
on:
  push:
    branches:
      - main
```

---

# 45. GitHub Actions Pipeline

The pipeline performs:

```text
Checkout code
      |
      v
Install Python
      |
      v
Install dependencies
      |
      v
Run data tests
      |
      v
Validate data
      |
      v
Train ML pipeline
      |
      v
Run prediction test
      |
      v
Authenticate to AWS
      |
      v
Login to ECR
      |
      v
Build Docker image
      |
      v
Push image to ECR
      |
      v
Update EKS kubeconfig
      |
      v
Update Kubernetes deployment
      |
      v
Wait for rollout
```

---

# 46. GitHub OIDC Authentication

### What are we doing?

Allowing GitHub Actions to authenticate to AWS without storing long-lived AWS access keys.

Architecture:

```text
GitHub Actions
      |
      | OIDC token
      v
GitHub OIDC Provider
      |
      v
AWS IAM Role
      |
      v
Temporary AWS Credentials
```

IAM role:

```text
GitHubActionsLoanApprovalRole
```

The trust policy restricts the role to the project repository and `main` branch.

---

# 47. EKS Access for GitHub Actions

The GitHub Actions IAM role is configured as an EKS access entry.

It receives:

```text
AmazonEKSEditPolicy
```

for:

```text
default namespace
```

This allows GitHub Actions to update the Kubernetes deployment.

---

# 48. ECR Permissions

The GitHub Actions IAM role requires permissions to:

```text
ecr:GetAuthorizationToken
ecr:BatchCheckLayerAvailability
ecr:CompleteLayerUpload
ecr:InitiateLayerUpload
ecr:PutImage
ecr:UploadLayerPart
```

These allow the workflow to authenticate and push Docker images.

---

# 49. EKS Permission

The workflow needs:

```text
eks:DescribeCluster
```

to configure the Kubernetes client against the EKS cluster.

Kubernetes deployment authorization is handled through EKS access configuration.

---

# 50. GitHub Actions Docker Build

The production workflow builds the image for AMD64:

```bash
docker build --platform linux/amd64   -t $ECR_REGISTRY/$ECR_REPOSITORY:${{ github.sha }}   -t $ECR_REGISTRY/$ECR_REPOSITORY:latest .
```

Two tags are created:

```text
commit SHA
latest
```

The commit SHA provides an immutable version of the image.

---

# 51. Push Image to ECR

```bash
docker push $ECR_REGISTRY/$ECR_REPOSITORY:${{ github.sha }}
docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
```

---

# 52. Deploy Image to EKS from GitHub Actions

First configure kubeconfig:

```bash
aws eks update-kubeconfig   --region $AWS_REGION   --name loan-approval-cluster
```

Then update the Kubernetes Deployment:

```bash
kubectl set image deployment/loan-approval-api   loan-approval-api=$ECR_REGISTRY/$ECR_REPOSITORY:${{ github.sha }}
```

The Deployment therefore runs the exact Docker image created by that GitHub Actions run.

---

# 53. Wait for Kubernetes Rollout

```bash
kubectl rollout status   deployment/loan-approval-api   --timeout=180s
```

The workflow only completes successfully when the new application version has rolled out successfully.

---

# 54. Complete GitHub Actions Workflow

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
  ECR_REGISTRY: 517811334575.dkr.ecr.ap-south-1.amazonaws.com
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
          role-to-assume: arn:aws:iam::517811334575:role/GitHubActionsLoanApprovalRole
          aws-region: ap-south-1

      - name: Login to Amazon ECR
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build Docker image
        run: |
          docker build --platform linux/amd64             -t $ECR_REGISTRY/$ECR_REPOSITORY:${{ github.sha }}             -t $ECR_REGISTRY/$ECR_REPOSITORY:latest .

      - name: Push Docker image to ECR
        run: |
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:${{ github.sha }}
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest

      - name: Update kubeconfig
        run: |
          aws eks update-kubeconfig             --region $AWS_REGION             --name loan-approval-cluster

      - name: Deploy to EKS
        run: |
          kubectl set image deployment/loan-approval-api             loan-approval-api=$ECR_REGISTRY/$ECR_REPOSITORY:${{ github.sha }}

      - name: Wait for deployment
        run: |
          kubectl rollout status deployment/loan-approval-api --timeout=180s
```

---

# 55. Final End-to-End Flow

```text
                    Git Push
                       |
                       v
                GitHub Repository
                       |
                       v
                GitHub Actions
                       |
                       v
              Install Dependencies
                       |
                       v
                Data Validation
                       |
                       v
                    Pytest
                       |
                       v
                 ML Training
                       |
                       v
              Hyperparameter Tuning
                       |
                       v
                 Best Model
                       |
                       v
               Model Evaluation
                       |
                       v
                MLflow Tracking
                       |
                       v
                Save Model
                       |
                       v
               Prediction Test
                       |
                       v
                AWS OIDC Login
                       |
                       v
                  ECR Login
                       |
                       v
                Docker Build
                       |
                       v
                  ECR Push
                       |
                       v
               Connect to EKS
                       |
                       v
             Update Deployment
                       |
                       v
              Kubernetes Rollout
                       |
                       v
               AWS Load Balancer
                       |
                       v
                   FastAPI
                       |
                       v
                  /predict
                       |
                       v
              Loan Prediction
```

---

# 56. Deployment Validation Checklist

## Local ML

```bash
python src/data_validation.py
```

```bash
pytest -v
```

Expected:

```text
5 passed
```

## Model

```bash
ls models/
```

Expected:

```text
loan_approval_model.pkl
```

## FastAPI

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Docker

```bash
docker build -t loan-approval-api:local .
```

```bash
docker run -p 8000:8000 loan-approval-api:local
```

## EKS

```bash
kubectl get nodes
```

Expected node status:

```text
Ready
```

```bash
kubectl get pods
```

Expected:

```text
1/1 Running
```

```bash
kubectl get svc
```

Expected:

```text
LoadBalancer
```

## GitHub Actions

The completed GitHub Actions deployment successfully executed:

```text
Data tests             ✓
Data validation        ✓
ML training            ✓
Prediction test        ✓
AWS authentication     ✓
ECR login              ✓
Docker build           ✓
ECR push               ✓
EKS kubeconfig         ✓
Kubernetes deployment  ✓
Rollout                ✓
```

---

# 57. Final Result

The project successfully demonstrates:

```text
Machine Learning
      +
Data Validation
      +
Automated Testing
      +
Hyperparameter Tuning
      +
MLflow
      +
FastAPI
      +
Docker
      +
Amazon ECR
      +
Amazon EKS
      +
Kubernetes
      +
GitHub Actions
      +
AWS OIDC/IAM
      =
End-to-End MLOps Pipeline
```

The deployed application successfully served a production prediction:

```json
{
  "prediction": "APPROVED",
  "approval_probability": 78.47
}
```

The complete path is:

```text
Dataset
  -> Validation
  -> Testing
  -> Training
  -> Tuning
  -> Evaluation
  -> MLflow
  -> Model Artifact
  -> FastAPI
  -> Docker
  -> ECR
  -> EKS
  -> Load Balancer
  -> Production API
```

This completes the implementation and deployment of the Loan Approval MLOps project.
