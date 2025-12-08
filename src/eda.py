"""
Exploratory Data Analysis (EDA) for Employee Attrition Dataset

Performs comprehensive EDA including:
    - Dataset overview and statistics
    - Target variable distribution
    - Feature correlations with target
    - Visualizations (distributions, correlations, boxplots)

Author: Souhaib MADHOUR
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import chi2_contingency


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Charge le dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset introuvable: {path}")
    return pd.read_csv(path)


# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------
def dataset_overview(df: pd.DataFrame):
    """Affiche un aperçu général du dataset."""
    print("="*70)
    print("APERÇU GÉNÉRAL DU DATASET")
    print("="*70)
    print(f"\nDimensions: {df.shape[0]} lignes × {df.shape[1]} colonnes")
    
    print(f"\nTypes de données:")
    print(df.dtypes.value_counts())
    
    print(f"\nMémoire utilisée: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n" + "-"*70)
    print("Aperçu des premières lignes:")
    print("-"*70)
    print(df.head())
    
    print("\n" + "-"*70)
    print("Statistiques descriptives (variables numériques):")
    print("-"*70)
    print(df.describe().T)


# --------------------------------------------------
# TARGET ANALYSIS
# --------------------------------------------------
def analyze_target(df: pd.DataFrame, save_dir: str = "../reports"):
    """Analyse la variable cible (Attrition)."""
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("ANALYSE DE LA VARIABLE CIBLE (ATTRITION)")
    print("="*70)
    
    attrition_counts = df['Attrition'].value_counts()
    attrition_pct = df['Attrition'].value_counts(normalize=True) * 100
    
    print(f"\nDistribution:")
    print(f"  No (Stayed):  {attrition_counts['No']:4d} ({attrition_pct['No']:.2f}%)")
    print(f"  Yes (Left):   {attrition_counts['Yes']:4d} ({attrition_pct['Yes']:.2f}%)")
    print(f"\nRatio Stayed:Left = {attrition_counts['No']/attrition_counts['Yes']:.2f}:1")
    print(f"\n⚠ Dataset DÉSÉQUILIBRÉ - Nécessite class_weight='balanced'")
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot
    attrition_counts.plot(kind='bar', ax=axes[0], color=['steelblue', 'coral'])
    axes[0].set_title('Distribution de l\'Attrition (Comptes)', fontweight='bold')
    axes[0].set_xlabel('Attrition')
    axes[0].set_ylabel('Nombre d\'employés')
    axes[0].set_xticklabels(['No (Stayed)', 'Yes (Left)'], rotation=0)
    
    for i, v in enumerate(attrition_counts):
        axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')
    
    # Pie chart
    axes[1].pie(attrition_counts, labels=['Stayed', 'Left'], autopct='%1.1f%%',
                colors=['steelblue', 'coral'], startangle=90)
    axes[1].set_title('Distribution de l\'Attrition (%)', fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'target_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualisation sauvegardée: {save_path}")


# --------------------------------------------------
# CORRELATION ANALYSIS
# --------------------------------------------------
def analyze_correlations(df: pd.DataFrame, save_dir: str = "../reports"):
    """Analyse les corrélations avec la variable cible."""
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("ANALYSE DES CORRÉLATIONS AVEC ATTRITION")
    print("="*70)
    
    # Encoder target
    df_copy = df.copy()
    df_copy['Attrition'] = df_copy['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Sélectionner seulement les colonnes numériques
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    # Calculer corrélations
    correlations = df_copy[numeric_cols].corr()['Attrition'].drop('Attrition').sort_values(ascending=False)
    
    print("\nTop 10 corrélations positives (facteurs d'attrition):")
    print(correlations.head(10))
    
    print("\nTop 10 corrélations négatives (facteurs de rétention):")
    print(correlations.tail(10))
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top positive correlations
    top_pos = correlations.head(10)
    axes[0].barh(range(len(top_pos)), top_pos.values, color='coral')
    axes[0].set_yticks(range(len(top_pos)))
    axes[0].set_yticklabels(top_pos.index)
    axes[0].set_xlabel('Corrélation avec Attrition')
    axes[0].set_title('Top 10 Facteurs d\'Attrition (Corrélation Positive)', fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Top negative correlations
    top_neg = correlations.tail(10).sort_values()
    axes[1].barh(range(len(top_neg)), top_neg.values, color='steelblue')
    axes[1].set_yticks(range(len(top_neg)))
    axes[1].set_yticklabels(top_neg.index)
    axes[1].set_xlabel('Corrélation avec Attrition')
    axes[1].set_title('Top 10 Facteurs de Rétention (Corrélation Négative)', fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'correlations_with_attrition.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualisation sauvegardée: {save_path}")
    
    # Correlation heatmap (top features)
    top_features = list(correlations.abs().nlargest(15).index) + ['Attrition']
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_copy[top_features].corr(), annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True, linewidths=1)
    plt.title('Heatmap des Corrélations (Top 15 Features)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'correlation_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Heatmap sauvegardée: {save_path}")


# --------------------------------------------------
# CATEGORICAL FEATURE ANALYSIS
# --------------------------------------------------
def analyze_categorical_features(df: pd.DataFrame, save_dir: str = "../reports"):
    """Analyse les features catégorielles en relation avec Attrition."""
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("ANALYSE DES VARIABLES CATÉGORIELLES")
    print("="*70)
    
    categorical_features = [
        'BusinessTravel', 'Department', 'EducationField', 
        'Gender', 'JobRole', 'MaritalStatus', 'OverTime'
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.ravel()
    
    chi2_results = {}
    
    for idx, feature in enumerate(categorical_features):
        if feature not in df.columns:
            continue
        
        # Crosstab
        ct = pd.crosstab(df[feature], df['Attrition'], normalize='index') * 100
        
        # Chi-square test
        contingency_table = pd.crosstab(df[feature], df['Attrition'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        chi2_results[feature] = {'chi2': chi2, 'p_value': p_value}
        
        # Plot
        ct.plot(kind='bar', ax=axes[idx], color=['steelblue', 'coral'], width=0.8)
        axes[idx].set_title(f'{feature}\n(χ² = {chi2:.2f}, p = {p_value:.4f})', fontweight='bold')
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('Pourcentage (%)')
        axes[idx].legend(['Stayed', 'Left'], loc='best')
        axes[idx].tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for idx in range(len(categorical_features), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'categorical_features_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Analyse catégorielle sauvegardée: {save_path}")
    
    # Print chi-square results
    print("\nTests Chi-carré (relation avec Attrition):")
    print(f"{'Feature':<25} {'Chi²':>10} {'p-value':>12} {'Significatif':>15}")
    print("-"*70)
    for feature, results in sorted(chi2_results.items(), key=lambda x: x[1]['chi2'], reverse=True):
        sig = "✓ Oui" if results['p_value'] < 0.05 else "✗ Non"
        print(f"{feature:<25} {results['chi2']:>10.2f} {results['p_value']:>12.4f} {sig:>15}")


# --------------------------------------------------
# NUMERICAL FEATURE DISTRIBUTIONS
# --------------------------------------------------
def analyze_numerical_distributions(df: pd.DataFrame, save_dir: str = "../reports"):
    """Analyse les distributions des features numériques."""
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("ANALYSE DES DISTRIBUTIONS (FEATURES NUMÉRIQUES)")
    print("="*70)
    
    numerical_features = [
        'Age', 'MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears',
        'DistanceFromHome', 'YearsInCurrentRole'
    ]
    
    df_copy = df.copy()
    df_copy['Attrition_num'] = df_copy['Attrition'].map({'Yes': 1, 'No': 0})
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    for idx, feature in enumerate(numerical_features):
        if feature not in df.columns:
            continue
        
        # Boxplot by Attrition
        df_copy.boxplot(column=feature, by='Attrition', ax=axes[idx])
        axes[idx].set_title(f'Distribution de {feature} par Attrition', fontweight='bold')
        axes[idx].set_xlabel('Attrition')
        axes[idx].set_ylabel(feature)
        plt.sca(axes[idx])
        plt.xticks([1, 2], ['No (Stayed)', 'Yes (Left)'])
    
    plt.suptitle('')  # Remove default title
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'numerical_distributions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Distributions numériques sauvegardées: {save_path}")


# --------------------------------------------------
# GENERATE FULL EDA REPORT
# --------------------------------------------------
def generate_eda_report(df: pd.DataFrame, save_dir: str = "../reports"):
    """Génère un rapport EDA complet en format texte."""
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, "eda_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("RAPPORT D'ANALYSE EXPLORATOIRE DES DONNÉES (EDA)\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Dataset: Employee Attrition\n")
        f.write(f"Dimensions: {df.shape[0]} lignes × {df.shape[1]} colonnes\n\n")
        
        f.write("-"*70 + "\n")
        f.write("1. DISTRIBUTION DE LA VARIABLE CIBLE\n")
        f.write("-"*70 + "\n")
        attrition_counts = df['Attrition'].value_counts()
        f.write(f"No (Stayed): {attrition_counts['No']} ({attrition_counts['No']/len(df)*100:.2f}%)\n")
        f.write(f"Yes (Left):  {attrition_counts['Yes']} ({attrition_counts['Yes']/len(df)*100:.2f}%)\n")
        f.write(f"Ratio: {attrition_counts['No']/attrition_counts['Yes']:.2f}:1\n\n")
        
        f.write("-"*70 + "\n")
        f.write("2. STATISTIQUES DESCRIPTIVES (TOP FEATURES NUMÉRIQUES)\n")
        f.write("-"*70 + "\n")
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]
        f.write(df[numeric_cols].describe().to_string())
        f.write("\n\n")
        
        f.write("-"*70 + "\n")
        f.write("3. VALEURS MANQUANTES\n")
        f.write("-"*70 + "\n")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            f.write("✓ Aucune valeur manquante détectée\n\n")
        else:
            f.write(missing[missing > 0].to_string())
            f.write("\n\n")
        
        f.write("-"*70 + "\n")
        f.write("4. RECOMMANDATIONS\n")
        f.write("-"*70 + "\n")
        f.write("- Dataset déséquilibré → Utiliser class_weight='balanced'\n")
        f.write("- Analyser les corrélations pour feature engineering\n")
        f.write("- Considérer les interactions entre features\n")
        f.write("- Tester plusieurs algorithmes (RF, XGBoost, LightGBM)\n\n")
        
        f.write("="*70 + "\n")
        f.write("FIN DU RAPPORT EDA\n")
        f.write("="*70 + "\n")
    
    print(f"✓ Rapport EDA sauvegardé: {report_path}")


# --------------------------------------------------
# MAIN EDA PIPELINE
# --------------------------------------------------
def main():
    """
    Pipeline EDA complet:
        1. Aperçu du dataset
        2. Analyse de la cible
        3. Corrélations
        4. Features catégorielles
        5. Distributions numériques
        6. Rapport final
    """
    
    print("="*70)
    print("ANALYSE EXPLORATOIRE DES DONNÉES (EDA)")
    print("="*70)
    
    # Load data
    data_path = "../data/employee_attrition.csv"
    df = load_data(data_path)
    
    # Dataset overview
    dataset_overview(df)
    
    # Target analysis
    analyze_target(df)
    
    # Correlation analysis
    analyze_correlations(df)
    
    # Categorical features
    analyze_categorical_features(df)
    
    # Numerical distributions
    analyze_numerical_distributions(df)
    
    # Generate report
    generate_eda_report(df)
    
    print("\n" + "="*70)
    print("EDA TERMINÉE!")
    print("="*70)
    print("\nTous les résultats sont disponibles dans le dossier 'reports/':")
    print("  - target_distribution.png")
    print("  - correlations_with_attrition.png")
    print("  - correlation_heatmap.png")
    print("  - categorical_features_analysis.png")
    print("  - numerical_distributions.png")
    print("  - eda_report.txt")


if __name__ == "__main__":
    main()