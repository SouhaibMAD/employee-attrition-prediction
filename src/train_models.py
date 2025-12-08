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
    accuracy_score,
    roc_curve,
    auc
)
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import joblib
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import preprocessing
from preprocessing import preprocess_data


# --------------------------------------------------
# VISUALIZATION FUNCTIONS
# --------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, model_name, save_dir="../reports"):
    """
    Crée et sauvegarde une matrice de confusion visualisée.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Stayed (0)', 'Left (1)'], 
                yticklabels=['Stayed (0)', 'Left (1)'],
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Matrice de Confusion - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Vraie Classe', fontsize=12)
    plt.xlabel('Classe Prédite', fontsize=12)
    
    # Ajouter les métriques sur le graphique
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    textstr = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}'
    plt.text(1.5, -0.15, textstr, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Matrice de confusion sauvegardée: {save_path}")


def plot_roc_curves(models_dict, X_val, y_val, save_dir="../reports"):
    """
    Crée et sauvegarde les courbes ROC pour tous les modèles.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for idx, (name, model) in enumerate(models_dict.items()):
        y_proba = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2,
                label=f'{name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5000)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (FPR)', fontsize=12)
    plt.ylabel('Taux de Vrais Positifs (TPR)', fontsize=12)
    plt.title('Courbes ROC - Comparaison des Modèles', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'roc_curves_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Courbes ROC sauvegardées: {save_path}")


def plot_feature_importance(model, feature_names, model_name, top_n=15, save_dir="../reports"):
    """
    Crée et sauvegarde un graphique des features les plus importantes.
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"  ⚠ {model_name} ne supporte pas feature_importances_")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices], color='steelblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Features Importantes - {model_name}', 
              fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Feature importance sauvegardée: {save_path}")
    
    # Afficher aussi dans la console
    print(f"\n  Top {min(10, top_n)} features importantes ({model_name}):")
    for i in indices[::-1][:10]:
        print(f"    {feature_names[i]:30s}: {importances[i]:.4f}")


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
    
    # Save comparison table
    os.makedirs("../reports", exist_ok=True)
    comparison_df.to_csv("reports/baseline_comparison.csv")
    print("\n✓ Tableau de comparaison sauvegardé: reports/baseline_comparison.csv")
    
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
        'n_estimators': [100, 150],  # Réduit
        'max_depth': [10, 15, 20],   # Enlève "None", ajoute limite
        'min_samples_split': [5, 10, 20],  # Augmente les valeurs
        'min_samples_leaf': [2, 4, 8],     # Augmente les valeurs
        'class_weight': ['balanced']
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=5,
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
        'n_estimators': [100, 150],
        'max_depth': [3, 5],           # Réduit la profondeur (3,5 au lieu de 3,5,7)
        'learning_rate': [0.01, 0.1],
        'subsample': [0.7, 0.8],       # Réduit de 0.8-1.0 à 0.7-0.8
        'colsample_bytree': [0.7, 0.8], # Réduit de 0.8-1.0 à 0.7-0.8
        'reg_alpha': [0, 0.1],          # AJOUTÉ: régularisation L1
        'reg_lambda': [1, 2]           # AJOUTÉ: régularisation L2
    }
    
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=10  # ← AJOUTÉ
    )
    
    grid_search = GridSearchCV(
        xgb_model,
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    print("Running grid search (this may take a few minutes)...")
    grid_search.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],  # ← AJOUTÉ
        verbose=False
    )
    
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
    data_path = "../data/employee_attrition.csv"
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, feature_names = preprocess_data(data_path)
    
    # Step 1: Train baseline models
    baseline_models, baseline_results = train_baseline_models(X_train, y_train, X_val, y_val)
    
    # Create visualizations for baseline models
    print("\n" + "="*60)
    print("CRÉATION DES VISUALISATIONS (BASELINE)")
    print("="*60)
    plot_roc_curves(baseline_models, X_val, y_val)
    
    for name, model in baseline_models.items():
        y_pred = model.predict(X_val)
        plot_confusion_matrix(y_val, y_pred, name)
    
    # Step 2: Hyperparameter tuning for top 2 models
    print("\n" + "="*60)
    print("STEP 2: HYPERPARAMETER TUNING")
    print("="*60)
    
    # Tune Random Forest
    best_rf, rf_params, rf_metrics = tune_random_forest(X_train, y_train, X_val, y_val)
    save_model_and_results(best_rf, "Random_Forest", rf_params, rf_metrics)
    
    print("\nGénération des visualisations (Random Forest)...")
    y_pred_rf = best_rf.predict(X_val)
    plot_confusion_matrix(y_val, y_pred_rf, "Random_Forest_Tuned")
    plot_feature_importance(best_rf, feature_names, "Random_Forest", top_n=15)
    
    # Tune XGBoost
    best_xgb, xgb_params, xgb_metrics = tune_xgboost(X_train, y_train, X_val, y_val)
    save_model_and_results(best_xgb, "XGBoost", xgb_params, xgb_metrics)
    
    print("\nGénération des visualisations (XGBoost)...")
    y_pred_xgb = best_xgb.predict(X_val)
    plot_confusion_matrix(y_val, y_pred_xgb, "XGBoost_Tuned")
    plot_feature_importance(best_xgb, feature_names, "XGBoost", top_n=15)
    
    # Plot ROC for tuned models
    tuned_models = {
        "Random Forest (Tuned)": best_rf,
        "XGBoost (Tuned)": best_xgb
    }
    plot_roc_curves(tuned_models, X_val, y_val, save_dir="../reports")
    
    # Step 3: Select best model
    print("\n" + "="*60)
    print("FINAL MODEL SELECTION")
    print("="*60)
    
    if rf_metrics['roc_auc'] > xgb_metrics['roc_auc']:
        best_model = best_rf
        best_name = "Random Forest"
        best_metrics = rf_metrics
        print(f"✓ Best model: Random Forest (ROC-AUC: {rf_metrics['roc_auc']:.4f})")
    else:
        best_model = best_xgb
        best_name = "XGBoost"
        best_metrics = xgb_metrics
        print(f"✓ Best model: XGBoost (ROC-AUC: {xgb_metrics['roc_auc']:.4f})")
    
    # Save as "best_model"
    joblib.dump(best_model, "models/best_model.pkl")
    print(f"✓ Best model saved as: models/best_model.pkl")
    
    # Save best model info
    best_model_info = {
        "model_name": best_name,
        "metrics": best_metrics,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open("models/best_model_info.json", 'w') as f:
        json.dump(best_model_info, f, indent=4)
    print(f"✓ Best model info saved: models/best_model_info.json")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"✓ Tous les modèles et visualisations sont sauvegardés")
    print(f"✓ Next step: Run evaluate_model.py to test on the test set")


if __name__ == "__main__":
    main()