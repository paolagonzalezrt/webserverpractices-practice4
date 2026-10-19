from faker import Faker
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# STEP 1: Configure random seeds
fake = Faker("es_MX")
Faker.seed(42)
np.random.seed(42)

print("STEP 1: RANDOM SEEDS CONFIGURED")
print("Faker locale: es_MX")
print("Random seed: 42")

input("\nPress Enter to continue...")

# STEP 2: Generate 1,000 synthetic customers
records = []

for i in range(1000):
    records.append({
        "customer_id": i + 1,
        "name": fake.name(),
        "email": fake.email(),
        "address": fake.address().replace("\n", ", "),
        "phone_number": fake.phone_number(),
        "marital_status": np.random.choice(
            ["Single", "Married", "Divorced"],
            p=[0.5, 0.4, 0.1]
        ),
        "annual_income": round(
            np.random.uniform(100000, 1000000), 2
        ),
        "churn": np.random.choice([0, 1], p=[0.8, 0.2])
    })

df_synthetic = pd.DataFrame(records)

print("\nSTEP 2: SYNTHETIC DATASET")
print(df_synthetic.head())
print(f"\nDataset dimensions: {df_synthetic.shape}")
print("\nChurn distribution:")
print(df_synthetic["churn"].value_counts().sort_index())

input("\nPress Enter to continue...")

# STEP 3: Separate features and target
X = df_synthetic.drop(
    columns=[
        "customer_id",
        "name",
        "email",
        "address",
        "phone_number",
        "churn"
    ]
)

y = df_synthetic["churn"]

print("\nSTEP 3: FEATURES AND TARGET")
print("Feature columns:", X.columns.tolist())
print("Feature dimensions:", X.shape)
print("Target dimensions:", y.shape)

input("\nPress Enter to continue...")

# STEP 4: Encode categorical variables
X = pd.get_dummies(
    X,
    columns=["marital_status"],
    dtype=int
)

print("\nSTEP 4: ENCODED FEATURES")
print(X.head())
print("Encoded feature dimensions:", X.shape)

input("\nPress Enter to continue...")

# STEP 5: Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nSTEP 5: STRATIFIED TRAIN-TEST SPLIT")
print("Training feature dimensions:", X_train.shape)
print("Testing feature dimensions:", X_test.shape)
print(f"Training churn ratio: {y_train.mean():.3f}")
print(f"Testing churn ratio: {y_test.mean():.3f}")

input("\nPress Enter to continue...")

# STEP 6: Compare churn class distributions
distribution = pd.DataFrame({
    "Full Dataset": y.value_counts(normalize=True),
    "Training Set": y_train.value_counts(normalize=True),
    "Test Set": y_test.value_counts(normalize=True)
}).sort_index()

distribution.index.name = "Churn Class"
distribution = distribution.rename(
    index={0: "No Churn (0)", 1: "Churn (1)"}
)

print("\nSTEP 6: CHURN CLASS DISTRIBUTION")
print((distribution * 100).round(2).astype(str) + "%")

input("\nPress Enter to continue...")

# STEP 7: Save datasets and results
df_synthetic.to_csv("synthetic_customers.csv", index=False)

training_data = X_train.copy()
training_data["churn"] = y_train
training_data.to_csv("train_data.csv", index=False)

testing_data = X_test.copy()
testing_data["churn"] = y_test
testing_data.to_csv("test_data.csv", index=False)

distribution.to_csv("churn_distribution.csv")

print("\nSTEP 7: FILES SAVED")
print("- synthetic_customers.csv")
print("- train_data.csv")
print("- test_data.csv")
print("- churn_distribution.csv")
print("\nPractice 4 completed successfully.")
