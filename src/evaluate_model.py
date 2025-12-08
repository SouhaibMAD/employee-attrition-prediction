"""
Final Model Evaluation on Test Set

Evaluates the best trained model on the test set and generates:
    - Performance metrics
    - Confusion matrix
    - ROC curve
    - Feature importance
    - Classification report

Author: Souhaib MADHOUR
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from preprocessing import preprocess_data


# --------------------------------------------------
# VISUALIZATION FUNCTIONS
# --------------------------------------------------
def plot_confusion_matrix_final(y_true, y_pred, save_dir="../reports"):
    """
    Crée une matrice de confusion détaillée pour le test final.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Matrice de confusion (counts)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Stayed (0)', 'Left (1)'], 
                yticklabels=['Stayed (0)', 'Left (1)'],
                cbar_kws={'label': 'Count'})
    ax1.set_title('Matrice de Confusion - Test Set', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Vraie Classe', fontsize=12)
    ax1.set_xlabel('Classe Prédite', fontsize=12)
    
    # Matrice de confusion (percentages)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens', ax=ax2,
                xticklabels=['Stayed (0)', 'Left (1)'], 
                yticklabels=['Stayed (0)', 'Left (1)'],
                cbar_kws={'label': 'Percentage (%)'})
    ax2.set_title('Matrice de Confusion (%) - Test Set', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Vraie Classe', fontsize=12)
    ax2.set_xlabel('Classe Prédite', fontsize=12)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'confusion_matrix_test_final.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Matrice de confusion finale sauvegardée: {save_path}")
    
    # Print detailed breakdown
    print("\n" + "="*60)
    print("ANALYSE DÉTAILLÉE DE LA MATRICE DE CONFUSION")
    print("="*60)
    print(f"True Negatives (TN):  {tn:4d} (Stayed correctement prédits)")
    print(f"False Positives (FP): {fp:4d} (Stayed prédits comme Left)")
    print(f"False Negatives (FN): {fn:4d} (Left prédits comme Stayed)")
    print(f"True Positives (TP):  {tp:4d} (Left correctement prédits)")
    print(f"\nTaux de Stayed correctement classés: {tn/(tn+fp)*100:.1f}%")
    print(f"Taux de Left correctement classés:   {tp/(tp+fn)*100:.1f}%")


def plot_roc_curve_final(y_true, y_proba, model_name, save_dir="../reports"):
    """
    Crée la courbe ROC pour l'évaluation finale.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier (AUC = 0.5000)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (FPR)', fontsize=12)
    plt.ylabel('Taux de Vrais Positifs (TPR)', fontsize=12)
    plt.title(f'Courbe ROC - Test Set Final', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'roc_curve_test_final.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Courbe ROC finale sauvegardée: {save_path}")


def plot_metrics_comparison(train_metrics, val_metrics, test_metrics, save_dir="../reports"):
    """
    Compare les métriques sur les 3 ensembles (train, val, test).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    train_vals = [train_metrics.get(m, 0) for m in metrics_names]
    val_vals = [val_metrics.get(m, 0) for m in metrics_names]
    test_vals = [test_metrics.get(m, 0) for m in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, train_vals, width, label='Train', color='skyblue')
    ax.bar(x, val_vals, width, label='Validation', color='lightcoral')
    ax.bar(x + width, test_vals, width, label='Test', color='lightgreen')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Comparaison des Métriques (Train/Val/Test)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_names])
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for i, (t, v, te) in enumerate(zip(train_vals, val_vals, test_vals)):
        ax.text(i - width, t + 0.02, f'{t:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width, te + 0.02, f'{te:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'metrics_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparaison des métriques sauvegardée: {save_path}")


# --------------------------------------------------
# EVALUATION FUNCTIONS
# --------------------------------------------------
def evaluate_model(model, X, y):
    """
    Évalue le modèle et retourne toutes les métriques.
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
    
    return metrics, y_pred, y_proba


def print_classification_report(y_true, y_pred):
    """
    Affiche un rapport de classification détaillé.
    """
    print("\n" + "="*60)
    print("RAPPORT DE CLASSIFICATION DÉTAILLÉ")
    print("="*60)
    print(classification_report(y_true, y_pred, 
                                target_names=['Stayed (0)', 'Left (1)'],
                                digits=4))


def save_final_report(model_info, train_metrics, val_metrics, test_metrics, save_dir="../reports"):
    """
    Sauvegarde un rapport final complet en format texte.
    """
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, "final_evaluation_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("RAPPORT D'ÉVALUATION FINALE - PRÉDICTION D'ATTRITION\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Modèle: {model_info.get('model_name', 'N/A')}\n")
        f.write(f"Date d'entraînement: {model_info.get('training_date', 'N/A')}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("MÉTRIQUES DE PERFORMANCE\n")
        f.write("-"*70 + "\n\n")
        
        f.write(f"{'Métrique':<20} {'Train':>12} {'Validation':>12} {'Test':>12}\n")
        f.write("-"*70 + "\n")
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in metrics:
            f.write(f"{metric.replace('_', ' ').title():<20} ")
            f.write(f"{train_metrics.get(metric, 0):>12.4f} ")
            f.write(f"{val_metrics.get(metric, 0):>12.4f} ")
            f.write(f"{test_metrics.get(metric, 0):>12.4f}\n")
        
        f.write("\n" + "-"*70 + "\n")
        f.write("ANALYSE\n")
        f.write("-"*70 + "\n\n")
        
        # Check for overfitting
        train_test_gap = train_metrics['roc_auc'] - test_metrics['roc_auc']
        if train_test_gap > 0.1:
            f.write("⚠ ATTENTION: Signe potentiel d'overfitting détecté\n")
            f.write(f"   Écart Train-Test ROC-AUC: {train_test_gap:.4f}\n\n")
        else:
            f.write("✓ Pas d'overfitting significatif détecté\n")
            f.write(f"   Écart Train-Test ROC-AUC: {train_test_gap:.4f}\n\n")
        
        # Performance assessment
        test_auc = test_metrics['roc_auc']
        if test_auc >= 0.9:
            f.write("✓ Performance EXCELLENTE (ROC-AUC ≥ 0.90)\n")
        elif test_auc >= 0.8:
            f.write("✓ Performance BONNE (ROC-AUC ≥ 0.80)\n")
        elif test_auc >= 0.7:
            f.write("○ Performance ACCEPTABLE (ROC-AUC ≥ 0.70)\n")
        else:
            f.write("⚠ Performance INSUFFISANTE (ROC-AUC < 0.70)\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("FIN DU RAPPORT\n")
        f.write("="*70 + "\n")
    
    print(f"✓ Rapport final sauvegardé: {report_path}")


# --------------------------------------------------
# MAIN EVALUATION PIPELINE
# --------------------------------------------------
def main():
    """
    Pipeline d'évaluation finale:
        1. Charger le meilleur modèle
        2. Évaluer sur test set
        3. Générer toutes les visualisations
        4. Sauvegarder le rapport final
    """
    
    print("="*60)
    print("ÉVALUATION FINALE SUR TEST SET")
    print("="*60)
    
    # Load best model
    print("\nChargement du meilleur modèle...")
    if not os.path.exists("models/best_model.pkl"):
        print("❌ Erreur: models/best_model.pkl introuvable!")
        print("   Veuillez d'abord exécuter train_models.py")
        return
    
    model = joblib.load("models/best_model.pkl")
    print("✓ Modèle chargé avec succès")
    
    # Load model info
    model_info = {}
    if os.path.exists("models/best_model_info.json"):
        with open("models/best_model_info.json", 'r') as f:
            model_info = json.load(f)
    
    model_name = model_info.get('model_name', 'Best Model')
    print(f"✓ Modèle: {model_name}")
    
    # Load preprocessed data
    print("\nChargement des données...")
    data_path = "../data/employee_attrition.csv"
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, feature_names = preprocess_data(data_path)
    
    # Evaluate on all sets
    print("\n" + "="*60)
    print("ÉVALUATION SUR LES 3 ENSEMBLES")
    print("="*60)
    
    print("\n[1/3] Évaluation sur TRAIN SET...")
    train_metrics, _, _ = evaluate_model(model, X_train, y_train)
    print(f"  ROC-AUC: {train_metrics['roc_auc']:.4f}")
    
    print("\n[2/3] Évaluation sur VALIDATION SET...")
    val_metrics, _, _ = evaluate_model(model, X_val, y_val)
    print(f"  ROC-AUC: {val_metrics['roc_auc']:.4f}")
    
    print("\n[3/3] Évaluation sur TEST SET...")
    test_metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test)
    
    # Display test metrics
    print("\n" + "="*60)
    print("MÉTRIQUES FINALES (TEST SET)")
    print("="*60)
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    
    # Classification report
    print_classification_report(y_test, y_pred)
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("="*60)
    
    plot_confusion_matrix_final(y_test, y_pred)
    plot_roc_curve_final(y_test, y_proba, model_name)
    plot_metrics_comparison(train_metrics, val_metrics, test_metrics)
    
    # Save final metrics
    final_results = {
        "model_name": model_name,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "evaluation_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    os.makedirs("../reports", exist_ok=True)
    with open("../reports/final_test_results.json", 'w') as f:
        json.dump(final_results, f, indent=4)
    print("✓ Résultats finaux sauvegardés: reports/final_test_results.json")
    
    # Generate final report
    save_final_report(model_info, train_metrics, val_metrics, test_metrics)
    
    # Check for overfitting
    print("\n" + "="*60)
    print("ANALYSE DE GÉNÉRALISATION")
    print("="*60)
    train_test_gap = train_metrics['roc_auc'] - test_metrics['roc_auc']
    
    if train_test_gap > 0.1:
        print(f"⚠ ATTENTION: Overfitting potentiel détecté!")
        print(f"   Écart Train-Test ROC-AUC: {train_test_gap:.4f}")
    else:
        print(f"✓ Bonne généralisation du modèle")
        print(f"   Écart Train-Test ROC-AUC: {train_test_gap:.4f}")
    
    print("\n" + "="*60)
    print("ÉVALUATION TERMINÉE!")
    print("="*60)
    print("\nTous les résultats sont disponibles dans le dossier 'reports/':")
    print("  - confusion_matrix_test_final.png")
    print("  - roc_curve_test_final.png")
    print("  - metrics_comparison.png")
    print("  - final_evaluation_report.txt")
    print("  - final_test_results.json")


if __name__ == "__main__":
    main()