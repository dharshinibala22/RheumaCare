import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

data_path = "data/ra_patients_synthetic.xlsx"

df = pd.read_excel(data_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. DISPLAY BASIC INFORMATION
# ==========================================

print("\nDiagnosis distribution:")
print(df["Diagnosis"].value_counts())


# ==========================================
# 3. SELECT FEATURES
# ==========================================

features = [
    "Age",
    "Sex",
    "SmokingStatus",
    "FamilyHistoryRA",
    "RF_IU_mL",
    "AntiCCP_U_mL",
    "ESR_mm_hr",
    "CRP_mg_L",
    "TenderJointCount",
    "SwollenJointCount",
    "MorningStiffness_min",
    "DiseaseDuration_months"
]

target = "Diagnosis"


X = df[features]
y = df[target]


# ==========================================
# 4. CATEGORICAL FEATURES
# ==========================================

categorical_features = [
    "Sex",
    "SmokingStatus",
    "FamilyHistoryRA"
]


# ==========================================
# 5. NUMERICAL FEATURES
# ==========================================

numerical_features = [
    "Age",
    "RF_IU_mL",
    "AntiCCP_U_mL",
    "ESR_mm_hr",
    "CRP_mg_L",
    "TenderJointCount",
    "SwollenJointCount",
    "MorningStiffness_min",
    "DiseaseDuration_months"
]


# ==========================================
# 6. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ==========================================
# 7. RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# ==========================================
# 8. CREATE PIPELINE
# ==========================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ==========================================
# 9. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 10. TRAIN MODEL
# ==========================================

print("\nTraining Random Forest model...")

pipeline.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 11. PREDICTION
# ==========================================

y_pred = pipeline.predict(X_test)


# ==========================================
# 12. EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    pos_label="RA-Positive"
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label="RA-Positive"
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label="RA-Positive"
)


print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1 Score :", round(f1 * 100, 2), "%")


# ==========================================
# 13. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# 14. CONFUSION MATRIX
# ==========================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==========================================
# 15. SAVE MODEL
# ==========================================

model_path = "model/ra_model.pkl"

joblib.dump(
    pipeline,
    model_path
)

print("\nModel saved successfully!")
print("Saved as:", model_path)