import pandas as pd


# Location of our dataset
DATA_PATH = "data/loan_approval.csv"


# Load the dataset
df = pd.read_csv(DATA_PATH)


print("===== DATASET SHAPE =====")
print(df.shape)


print("\n===== COLUMNS =====")
print(df.columns.tolist())


print("\n===== FIRST 5 ROWS =====")
print(df.head())


print("\n===== DATA TYPES =====")
print(df.dtypes)


print("\n===== MISSING VALUES =====")
print(df.isnull().sum())


print("\n===== TARGET DISTRIBUTION =====")
print(df["Loan_Status"].value_counts())
