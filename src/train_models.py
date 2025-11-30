"""
Model Training Script for Employee Attrition Prediction

Trains multiple ML models with hyperparameter tuning:
    - Logistic Regression
    - Random Forest
    - XGBoost
    - LightGBM (optional)

Saves best models and performance metrics.

Author: Souhaib MADHOUR
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import joblib
import os
import json
from datetime import datetime

# Import preprocessing
from preprocessing import preprocess_data


# --------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------
def evaluate_model(model, X, y, set_name="Validation"):
    """
    Evaluate model performance with multiple metrics.
    
    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba)
    }
    
    print(f"\n{set_name} Set Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics


# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------
def train_baseline_models(X_train, y_train, X_val, y_val):
    """
    Train baseline models without hyperparameter tuning.
    Quick comparison to see which models work best.
    """
    
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, 
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric='logloss'
        )
    }
    
    results = {}
    trained_models = {}
    
    print("="*60)
    print("TRAINING BASELINE MODELS")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        val_metrics = evaluate_model(model, X_val, y_val, "Validation")
        
        results[name] = val_metrics
        trained_models[name] = model
    
    # Summary
    print("\n" + "="*60)
    print("BASELINE MODEL COMPARISON (Validation Set)")
    print("="*60)
    
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.sort_values('roc_auc', ascending=False)
    print(comparison_df.to_string())
    
    return trained_models, results


# --------------------------------------------------
# HYPERPARAMETER TUNING
# --------------------------------------------------
def tune_random_forest(X_train, y_train, X_val, y_val):
    """
    Hyperparameter tuning for Random Forest using GridSearchCV.
    """
    
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING: Random Forest")
    print("="*60)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    print("Running grid search (this may take a few minutes)...")
    grid_search.fit(X_train, y_train)
    
    print(f"\n✓ Best parameters: {grid_search.best_params_}")
    print(f"✓ Best CV score: {grid_search.best_score_:.4f}")
    
    # Evaluate on validation set
    best_rf = grid_search.best_estimator_
    val_metrics = evaluate_model(best_rf, X_val, y_val, "Validation")
    
    return best_rf, grid_search.best_params_, val_metrics


def tune_xgboost(X_train, y_train, X_val, y_val):
    """
    Hyperparameter tuning for XGBoost.
    """
    
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING: XGBoost")
    print("="*60)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss'
    )
    
    grid_search = GridSearchCV(
        xgb_model,
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    print("Running grid search (this may take a few minutes)...")
    grid_search.fit(X_train, y_train)
    
    print(f"\n✓ Best parameters: {grid_search.best_params_}")
    print(f"✓ Best CV score: {grid_search.best_score_:.4f}")
    
    # Evaluate on validation set
    best_xgb = grid_search.best_estimator_
    val_metrics = evaluate_model(best_xgb, X_val, y_val, "Validation")
    
    return best_xgb, grid_search.best_params_, val_metrics


# --------------------------------------------------
# SAVE MODELS AND RESULTS
# --------------------------------------------------
def save_model_and_results(model, model_name, params, metrics, save_dir="models"):
    """
    Save trained model, parameters, and metrics.
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}.pkl")
    joblib.dump(model, model_path)
    print(f"✓ Model saved: {model_path}")
    
    # Save hyperparameters
    params_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_params.json")
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"✓ Parameters saved: {params_path}")
    
    # Save metrics
    metrics_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✓ Metrics saved: {metrics_path}")


# --------------------------------------------------
# MAIN TRAINING PIPELINE
# --------------------------------------------------
def main():
    """
    Complete training pipeline:
        1. Load preprocessed data
        2. Train baseline models
        3. Hyperparameter tuning for best models
        4. Save final models
    """
    
    print("="*60)
    print("EMPLOYEE ATTRITION PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Load preprocessed data
    data_path = "data\employee_attrition.csv"
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, feature_names = preprocess_data(data_path)
    
    # Step 1: Train baseline models
    baseline_models, baseline_results = train_baseline_models(X_train, y_train, X_val, y_val)
    
    # Step 2: Hyperparameter tuning for top 2 models
    print("\n" + "="*60)
    print("STEP 2: HYPERPARAMETER TUNING")
    print("="*60)
    
    # Tune Random Forest
    best_rf, rf_params, rf_metrics = tune_random_forest(X_train, y_train, X_val, y_val)
    save_model_and_results(best_rf, "Random_Forest", rf_params, rf_metrics)
    
    # Tune XGBoost
    best_xgb, xgb_params, xgb_metrics = tune_xgboost(X_train, y_train, X_val, y_val)
    save_model_and_results(best_xgb, "XGBoost", xgb_params, xgb_metrics)
    
    # Step 3: Select best model
    print("\n" + "="*60)
    print("FINAL MODEL SELECTION")
    print("="*60)
    
    if rf_metrics['roc_auc'] > xgb_metrics['roc_auc']:
        best_model = best_rf
        best_name = "Random Forest"
        print(f"✓ Best model: Random Forest (ROC-AUC: {rf_metrics['roc_auc']:.4f})")
    else:
        best_model = best_xgb
        best_name = "XGBoost"
        print(f"✓ Best model: XGBoost (ROC-AUC: {xgb_metrics['roc_auc']:.4f})")
    
    # Save as "best_model"
    joblib.dump(best_model, "models/best_model.pkl")
    print(f"✓ Best model saved as: models/best_model.pkl")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Next step: Run evaluate_model.py to test on the test set")


if __name__ == "__main__":
    main()