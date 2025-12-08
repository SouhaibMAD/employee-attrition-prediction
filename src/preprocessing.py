"""
HR Analytics - Employee Attrition Preprocessing Script

This module performs leakage-free preprocessing:
    ✓ Train/Val/Test split BEFORE any preprocessing
    ✓ Missing value imputation (fitted on train only)
    ✓ Ordinal + One-hot encoding
    ✓ Numerical scaling (fitted on train only)
    ✓ Outlier detection and reporting
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
import matplotlib.pyplot as plt
import seaborn as sns


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
# OUTLIER DETECTION
# --------------------------------------------------
def detect_outliers(X_train: pd.DataFrame, numeric_features: list) -> dict:
    """
    Détecte les outliers avec la méthode IQR (Interquartile Range).
    
    Args:
        X_train: Training features DataFrame
        numeric_features: List of numeric column names
    
    Returns:
        Dictionary with outlier counts per feature
    """
    outliers_summary = {}
    
    print("\n" + "="*60)
    print("DÉTECTION DES VALEURS ABERRANTES (Méthode IQR)")
    print("="*60)
    
    for col in numeric_features:
        if col in X_train.columns:
            Q1 = X_train[col].quantile(0.25)
            Q3 = X_train[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((X_train[col] < lower_bound) | (X_train[col] > upper_bound)).sum()
            outliers_summary[col] = {
                'count': outliers,
                'percentage': (outliers / len(X_train)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
    
    # Afficher les top 5 features avec le plus d'outliers
    sorted_outliers = sorted(outliers_summary.items(), 
                            key=lambda x: x[1]['count'], 
                            reverse=True)
    
    print("\nTop 5 features avec le plus de valeurs aberrantes:")
    for col, info in sorted_outliers[:5]:
        print(f"  {col:25s}: {info['count']:3d} outliers ({info['percentage']:5.2f}%)")
    
    print(f"\nNote: Les outliers sont détectés mais CONSERVÉS dans le dataset.")
    print(f"      (Suppression potentiellement dommageable pour la prédiction)")
    
    return outliers_summary


# --------------------------------------------------
# MISSING VALUES ANALYSIS
# --------------------------------------------------
def analyze_missing_values(df: pd.DataFrame) -> None:
    """
    Analyse et affiche les valeurs manquantes dans le dataset.
    """
    print("\n" + "="*60)
    print("ANALYSE DES VALEURS MANQUANTES")
    print("="*60)
    
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Colonnes': missing.index,
        'Valeurs_Manquantes': missing.values,
        'Pourcentage': missing_percent.values
    })
    
    missing_df = missing_df[missing_df['Valeurs_Manquantes'] > 0].sort_values(
        'Valeurs_Manquantes', ascending=False
    )
    
    if len(missing_df) > 0:
        print("\nColonnes avec valeurs manquantes:")
        print(missing_df.to_string(index=False))
    else:
        print("\n✓ Aucune valeur manquante détectée dans le dataset!")


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
# SAVE PREPROCESSING SUMMARY
# --------------------------------------------------
def save_preprocessing_summary(df: pd.DataFrame, outliers_info: dict, save_dir: str = "../reports"):
    """
    Sauvegarde un résumé du preprocessing en format texte.
    """
    os.makedirs(save_dir, exist_ok=True)
    summary_path = os.path.join(save_dir, "preprocessing_summary.txt")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("RÉSUMÉ DU PREPROCESSING\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Taille du dataset: {df.shape[0]} lignes, {df.shape[1]} colonnes\n\n")
        
        f.write("VALEURS MANQUANTES:\n")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            f.write("  ✓ Aucune valeur manquante\n\n")
        else:
            for col, count in missing[missing > 0].items():
                f.write(f"  - {col}: {count} ({count/len(df)*100:.2f}%)\n")
            f.write("\n")
        
        f.write("VALEURS ABERRANTES (TOP 5):\n")
        sorted_outliers = sorted(outliers_info.items(), 
                                key=lambda x: x[1]['count'], 
                                reverse=True)
        for col, info in sorted_outliers[:5]:
            f.write(f"  - {col}: {info['count']} outliers ({info['percentage']:.2f}%)\n")
        f.write("\n")
        
        f.write("MÉTHODE DE PREPROCESSING:\n")
        f.write("  1. Split stratifié (60% train, 20% val, 20% test)\n")
        f.write("  2. Imputation des valeurs manquantes (mode/médiane)\n")
        f.write("  3. Encodage ordinal + standardisation (features ordinales)\n")
        f.write("  4. One-hot encoding (features nominales)\n")
        f.write("  5. Standardisation (features numériques)\n")
        f.write("  6. Pipeline fitted UNIQUEMENT sur train (évite data leakage)\n")
    
    print(f"✓ Résumé du preprocessing sauvegardé: {summary_path}")


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
    
    # Analyze missing values
    analyze_missing_values(df)
    
    # Separate features and target
    y = encode_target(df["Attrition"])
    X = df.drop("Attrition", axis=1)
    
    # Check class distribution
    print(f"\n" + "="*60)
    print("DISTRIBUTION DES CLASSES")
    print("="*60)
    print(f"  Attrition=1 (Left):   {(y==1).sum():4d} ({(y==1).mean()*100:.1f}%)")
    print(f"  Attrition=0 (Stayed): {(y==0).sum():4d} ({(y==0).mean()*100:.1f}%)")
    print(f"  Ratio: 1:{(y==0).sum()/(y==1).sum():.2f} (Stayed:Left)")
    
    # Split: 60% train, 40% temp
    print(f"\n" + "="*60)
    print("SPLIT DES DONNÉES (STRATIFIÉ)")
    print("="*60)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    
    # Split temp: 50% val, 50% test (20% each of original)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"✓ Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
    print(f"✓ Val:   {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.0f}%)")
    print(f"✓ Test:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.0f}%)")
    
    # Detect outliers (on training set only)
    _, _, numeric_features = get_feature_groups()
    outliers_info = detect_outliers(X_train, numeric_features)
    
    # Build preprocessing pipeline (fit on train only)
    print(f"\n" + "="*60)
    print("CONSTRUCTION DU PIPELINE DE PREPROCESSING")
    print("="*60)
    preprocessor = build_preprocessing_pipeline(X_train)
    
    # Transform all splits
    print(f"\nTransformation des datasets...")
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
    
    # Save preprocessing summary
    save_preprocessing_summary(df, outliers_info)
    
    return X_train_prep, X_val_prep, X_test_prep, y_train, y_val, y_test, preprocessor, feature_names


# --------------------------------------------------
# EXECUTION
# --------------------------------------------------
if __name__ == "__main__":
    # Example usage
    data_path = "../data/employee_attrition.csv"
    
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, feature_names = preprocess_data(data_path)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Ready for model training!")