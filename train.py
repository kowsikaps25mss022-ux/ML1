import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

data = pd.read_csv(r"C:\HB\Dataset\Heartbeat stability.csv")
data.columns = data.columns.str.strip()
data = data.dropna(subset=["stability"])

X = data.drop(columns=["stability"]).select_dtypes(include="number")
X = X.fillna(X.median())

y = pd.to_numeric(data["stability"], errors="coerce")
valid = y.notna()
X = X.loc[valid]
y = y.loc[valid].astype(int)

if set(y.unique()) != {0, 1}:
    raise ValueError("Target must contain both classes 0 and 1.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
).fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc * 100:.2f}%")
print(classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)

version = len([
    file for file in os.listdir("models")
    if file.endswith(".pkl")
]) + 1

path = f"models/v{version}.pkl"

with open(path, "wb") as f:
    pickle.dump({
        "model": model,
        "features": list(X.columns),
        "target": "target",
        "dataset": "heartbeat_data.csv",
        "accuracy": acc
    }, f)

print("Saved:", path)