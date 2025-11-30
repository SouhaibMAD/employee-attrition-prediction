"""
HR Analytics - Employee Attrition Preprocessing Script

This module performs leakage-free preprocessing:
    ✓ Train/Val/Test split BEFORE any preprocessing
    ✓ Missing value imputation (fitted on train only)
    ✓ Ordinal + One-hot encoding
    ✓ Numerical scaling (fitted on train only)
    ✓ Saves preprocessing pipeline for reproducibility

Author: Souhaib MADHOUR
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    """Load employee attrition dataset from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)


# --------------------------------------------------
# TARGET ENCODING
# --------------------------------------------------
def encode_target(y: pd.Series) -> pd.Series:
    """
    Encode Attrition target variable:
        'Yes' → 1 (employee left)
        'No'  → 0 (employee stayed)
    """
    return y.map({"Yes": 1, "No": 0}).astype(int)


# --------------------------------------------------
# FEATURE DEFINITIONS
# --------------------------------------------------
def get_feature_groups():
    """
    Define feature groups for preprocessing.
    Returns: (categorical_ordinal, categorical_nominal, numeric_features)
    """
    
    # Ordinal features (have meaningful order: 1=low, 4=high)
    categorical_ordinal = [
        "Education",                    # 1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor
        "JobLevel",                     # 1-5 hierarchy
        "EnvironmentSatisfaction",      # 1=Low, 4=Very High
        "JobSatisfaction",              # 1=Low, 4=Very High
        "PerformanceRating",            # 1=Low, 4=Outstanding
        "RelationshipSatisfaction",     # 1=Low, 4=Very High
        "WorkLifeBalance",              # 1=Bad, 4=Best
        "JobInvolvement",               # 1=Low, 4=Very High
        "StockOptionLevel"              # 0-3 levels
    ]
    
    # Nominal features (no inherent order)
    categorical_nominal = [
        "BusinessTravel",               # Travel_Rarely, Travel_Frequently, Non-Travel
        "Department",                   # Sales, R&D, HR
        "EducationField",               # Life Sciences, Medical, etc.
        "Gender",                       # Male, Female
        "JobRole",                      # Sales Executive, Manager, etc.
        "MaritalStatus",                # Single, Married, Divorced
        "OverTime"                      # Yes, No
    ]
    
    # Numerical features
    numeric_features = [
        "Age", "DailyRate", "DistanceFromHome", "HourlyRate",
        "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
        "PercentSalaryHike", "TotalWorkingYears",
        "TrainingTimesLastYear", "YearsAtCompany", "YearsInCurrentRole",
        "YearsSinceLastPromotion", "YearsWithCurrManager"
    ]
    
    return categorical_ordinal, categorical_nominal, numeric_features


# --------------------------------------------------
# PREPROCESSING PIPELINE
# --------------------------------------------------
def build_preprocessing_pipeline(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Build and fit preprocessing pipeline on training data ONLY.
    
    Pipeline includes:
        - Ordinal encoding + scaling for ordered categorical features
        - One-hot encoding for nominal categorical features
        - Median imputation + scaling for numerical features
    
    Args:
        X_train: Training features (DataFrame)
    
    Returns:
        Fitted ColumnTransformer pipeline
    """
    
    categorical_ordinal, categorical_nominal, numeric_features = get_feature_groups()
    
    # Determine ordinal categories dynamically from training data
    ordinal_categories = []
    for col in categorical_ordinal:
        unique_vals = sorted(X_train[col].dropna().unique())
        ordinal_categories.append([str(v) for v in unique_vals])
    
    # Ordinal transformer: impute → encode → scale
    ordinal_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=ordinal_categories, handle_unknown="use_encoded_value", unknown_value=-1)),
        ("scaler", StandardScaler())
    ])
    
    # Nominal transformer: impute → one-hot encode
    nominal_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # Numeric transformer: impute → scale
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Combine all transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal", ordinal_transformer, categorical_ordinal),
            ("nominal", nominal_transformer, categorical_nominal),
            ("numeric", numeric_transformer, numeric_features)
        ],
        remainder="drop"  # Drop unused columns (EmployeeNumber, etc.)
    )
    
    # Fit ONLY on training data (prevents data leakage)
    preprocessor.fit(X_train)
    
    return preprocessor


# --------------------------------------------------
# FEATURE NAME EXTRACTION
# --------------------------------------------------
def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Extract feature names after preprocessing.
    Useful for model interpretation and explainability.
    """
    feature_names = []
    
    for name, transformer, features in preprocessor.transformers_:
        if name == "ordinal":
            feature_names.extend(features)
        elif name == "nominal":
            # Get one-hot encoded feature names
            onehot = transformer.named_steps["onehot"]
            feature_names.extend(onehot.get_feature_names_out(features))
        elif name == "numeric":
            feature_names.extend(features)
    
    return feature_names


# --------------------------------------------------
# MAIN PREPROCESSING FUNCTION
# --------------------------------------------------
def preprocess_data(data_path: str, save_dir: str = "models") -> tuple:
    """
    Complete preprocessing pipeline with train/val/test split.
    
    Steps:
        1. Load data
        2. Split into train/val/test (60/20/20) with stratification
        3. Build preprocessing pipeline fitted ONLY on training data
        4. Transform all splits
        5. Save pipeline for reproducibility
    
    Args:
        data_path: Path to CSV file
        save_dir: Directory to save preprocessing artifacts
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, feature_names)
    """
    
    # Load data
    print(f"Loading data from: {data_path}")
    df = load_dataset(data_path)
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Separate features and target
    y = encode_target(df["Attrition"])
    X = df.drop("Attrition", axis=1)
    
    # Check class distribution
    print(f"\nClass distribution:")
    print(f"  Attrition=1 (Left): {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
    print(f"  Attrition=0 (Stayed): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
    
    # Split: 60% train, 40% temp
    print(f"\nSplitting data (60/20/20)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    
    # Split temp: 50% val, 50% test (20% each of original)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"✓ Train: {X_train.shape[0]} samples")
    print(f"✓ Val:   {X_val.shape[0]} samples")
    print(f"✓ Test:  {X_test.shape[0]} samples")
    
    # Build preprocessing pipeline (fit on train only)
    print(f"\nBuilding preprocessing pipeline...")
    preprocessor = build_preprocessing_pipeline(X_train)
    
    # Transform all splits
    print(f"Transforming datasets...")
    X_train_prep = preprocessor.transform(X_train)
    X_val_prep = preprocessor.transform(X_val)
    X_test_prep = preprocessor.transform(X_test)
    
    print(f"✓ Train shape after preprocessing: {X_train_prep.shape}")
    print(f"✓ Val shape after preprocessing:   {X_val_prep.shape}")
    print(f"✓ Test shape after preprocessing:  {X_test_prep.shape}")
    
    # Extract feature names
    feature_names = get_feature_names(preprocessor)
    print(f"✓ Total features after encoding: {len(feature_names)}")
    
    # Save pipeline
    os.makedirs(save_dir, exist_ok=True)
    pipeline_path = os.path.join(save_dir, "preprocessing_pipeline.pkl")
    joblib.dump(preprocessor, pipeline_path)
    print(f"✓ Preprocessing pipeline saved to: {pipeline_path}")
    
    # Save feature names
    feature_names_path = os.path.join(save_dir, "feature_names.pkl")
    joblib.dump(feature_names, feature_names_path)
    print(f"✓ Feature names saved to: {feature_names_path}")
    
    return X_train_prep, X_val_prep, X_test_prep, y_train, y_val, y_test, preprocessor, feature_names


# --------------------------------------------------
# EXECUTION
# --------------------------------------------------
if __name__ == "__main__":
    # Example usage
    data_path = "data\employee_attrition.csv"
    
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, feature_names = preprocess_data(data_path)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Ready for model training!")