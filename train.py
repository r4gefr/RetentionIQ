import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load Data
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Clean Data
df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df.dropna(inplace=True)

# Encode Target
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Feature Engineering
df["HighMonthlyCharge"] = (
    df["MonthlyCharges"] > 80
).astype(int)

df["LongTermCustomer"] = (
    df["tenure"] > 24
).astype(int)

# Split
X = df.drop("Churn", axis=1)
y = df["Churn"]

X = pd.get_dummies(X)

feature_columns = X.columns.tolist()

joblib.dump(
    feature_columns,
    "models/feature_columns.joblib"
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)

print(classification_report(y_test, preds))

print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))

# Save
joblib.dump(
    model,
    "models/churn_model.joblib"
)

print("Model saved successfully.")