import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ==========================================
# STEP 1: LOAD HB DATASET
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "Dataset" / "Heartbeat stability.csv"


if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

data = pd.read_csv(DATA_PATH)

# Clean column names
data.columns = data.columns.str.strip()

print("\nDataset loaded successfully!")
print("Dataset shape:", data.shape)
print("Dataset columns:", data.columns.tolist())

# ==========================================
# STEP 2: IDENTIFY TARGET COLUMN
# ==========================================

TARGET_COLUMN = "Prediction"

if TARGET_COLUMN not in data.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found. "
        f"Available columns: {data.columns.tolist()}"
    )

# Remove rows with missing target values
data = data.dropna(subset=[TARGET_COLUMN]).copy()

# Remove whitespace from target values
data[TARGET_COLUMN] = (
    data[TARGET_COLUMN].astype(str).str.strip()
)

print("\nTarget column:", TARGET_COLUMN)
print("Target classes:", data[TARGET_COLUMN].unique())

# ==========================================
# STEP 3: SEPARATE FEATURES AND TARGET
# ==========================================

X = data.drop(columns=[TARGET_COLUMN])

# Select numerical features only
X = X.select_dtypes(include="number")

y = data[TARGET_COLUMN]

if X.empty:
    raise ValueError("No numeric feature columns found.")

if y.nunique() < 2:
    raise ValueError(
        "At least two target classes are required."
    )

if y.value_counts().min() < 2:
    raise ValueError(
        "Each target class must contain at least 2 rows."
    )

print("\nFeatures:", X.columns.tolist())
print("Total records:", len(data))
print("Feature count:", X.shape[1])

# ==========================================
# STEP 4: SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ==========================================
# STEP 5: SET MLFLOW EXPERIMENT
# ==========================================

mlflow.set_experiment(
    "HB_heartbeat_stability_prediction"
)

# ==========================================
# STEP 6: COMPARE RANDOM FOREST MODELS
# ==========================================

tree_values = [50, 100, 200]

best_accuracy = -1
best_model = None
best_trees = None

for trees in tree_values:

    print("\nTraining Random Forest...")
    print("Number of trees:", trees)

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=trees,
                    random_state=42
                )
            )
        ]
    )

    with mlflow.start_run():

        # ==================================
        # TRAIN MODEL
        # ==================================

        model.fit(X_train, y_train)

        # ==================================
        # MAKE PREDICTIONS
        # ==================================

        y_pred = model.predict(X_test)

        # ==================================
        # CALCULATE PERFORMANCE METRICS
        # ==================================

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        # ==================================
        # LOG PARAMETERS
        # ==================================

        mlflow.log_param("project", "HB")
        mlflow.log_param(
            "model_name", "Random Forest"
        )
        mlflow.log_param("n_estimators", trees)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param(
            "target_column", TARGET_COLUMN
        )

        # ==================================
        # LOG METRICS
        # ==================================

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # ==================================
        # LOG MODEL IN MLFLOW inside t
        # ==================================

        mlflow.sklearn.log_model(
            sk_model=model,
            name="heartbeat_stability_model",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],
        )

        # ==================================
        # FIND BEST MODEL
        # ==================================

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_trees = trees

        # ==================================
        # DISPLAY RESULTS
        # ==================================

        print("\nTrees:", trees)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        print("\nClassification Report:")
        print(classification_report(
            y_test,
            y_pred,
            zero_division=0
        ))

        print("--------------------------------")

# ==========================================
# STEP 7: SAVE BEST MODEL
# ==========================================

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "heartbeat_model.pkl"

joblib.dump(
    {
        "model": best_model,
        "features": list(X.columns),
        "target": TARGET_COLUMN,
        "dataset": DATA_PATH.name,
        "accuracy": best_accuracy,
        "n_estimators": best_trees
    },
    MODEL_PATH
)

print("\nBest model saved successfully!")
print("Best number of trees:", best_trees)
print("Best accuracy:", best_accuracy)
print("Model location:", MODEL_PATH)

# ==========================================
# STEP 8: FINAL RESULT
# ==========================================

print("\n================================")
print("HB MLFLOW EXPERIMENTS COMPLETED!")
print("================================")