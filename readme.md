# 🎯 Prédiction de l'Attrition des Employés

**Mini-projet Machine Learning - 4ème année Informatique et Réseaux**

Projet de classification binaire pour prédire l'attrition (départ) des employés au sein d'une entreprise en utilisant des techniques d'apprentissage automatique supervisé.

---

## 📋 Table des matières

- [Description](#-description)
- [Structure du projet](#-structure-du-projet)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Méthodologie](#-méthodologie)
- [Résultats](#-résultats)
- [Auteur](#-auteur)

---

## 📖 Description

Ce projet implémente un pipeline complet de Machine Learning pour prédire l'attrition des employés :

- **Objectif** : Prédire si un employé va quitter l'entreprise (Attrition = Yes/No)
- **Type de problème** : Classification binaire
- **Dataset** : HR Analytics Employee Attrition Dataset
- **Algorithmes testés** : Logistic Regression, Random Forest, XGBoost

### 🎓 Concepts ML couverts

✅ **Preprocessing**
- Split stratifié (train/val/test)
- Gestion des valeurs manquantes
- Encodage (ordinal + one-hot)
- Normalisation/Standardisation
- Détection des outliers

✅ **Modélisation**
- Modèles supervisés (classification)
- Gestion du déséquilibre de classes
- Hyperparameter tuning (GridSearchCV)
- Cross-validation (k-fold)

✅ **Évaluation**
- Métriques multiples (Accuracy, Precision, Recall, F1, ROC-AUC)
- Matrices de confusion
- Courbes ROC
- Analyse overfitting/underfitting

---

## 📁 Structure du projet

```
HR_ANALYTICS/
├── data/
│   └── employee_attrition.csv          # Dataset
│
├── models/                              # Modèles entraînés
│   ├── preprocessing_pipeline.pkl      # Pipeline de preprocessing
│   ├── feature_names.pkl               # Noms des features
│   ├── best_model.pkl                  # Meilleur modèle
│   ├── best_model_info.json            # Infos du meilleur modèle
│   ├── random_forest.pkl               # Random Forest tuné
│   ├── xgboost.pkl                     # XGBoost tuné
│   └── *_params.json / *_metrics.json  # Hyperparamètres et métriques
│
├── reports/                             # Résultats et visualisations
│   ├── eda_report.txt                  # Rapport EDA
│   ├── preprocessing_summary.txt       # Résumé preprocessing
│   ├── final_evaluation_report.txt     # Rapport final
│   ├── baseline_comparison.csv         # Comparaison modèles
│   ├── confusion_matrix_*.png          # Matrices de confusion
│   ├── roc_curves_*.png                # Courbes ROC
│   ├── feature_importance_*.png        # Features importantes
│   ├── target_distribution.png         # Distribution cible
│   ├── correlations_with_attrition.png # Corrélations
│   └── metrics_comparison.png          # Comparaison métriques
│
├── src/
│   ├── preprocessing.py                # Script de preprocessing
│   ├── train_models.py                 # Script d'entraînement
│   ├── evaluate_model.py               # Script d'évaluation
│   └── eda.py                          # Analyse exploratoire
│
├── requirements.txt                     # Dépendances Python
└── README.md                            # Documentation
```

---

## 🚀 Installation

### 1. Cloner le repository

```bash
git clone <repository_url>
cd HR_ANALYTICS
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 💻 Utilisation

### Étape 1 : Analyse exploratoire (EDA)

```bash
cd src
python eda.py
```

**Génère :**
- Visualisations de la distribution des données
- Corrélations avec la variable cible
- Analyse des features catégorielles et numériques
- Rapport EDA complet

### Étape 2 : Preprocessing + Entraînement

```bash
python train_models.py
```

**Effectue :**
1. Split stratifié des données (60% train, 20% val, 20% test)
2. Preprocessing (imputation, encodage, scaling)
3. Entraînement de modèles baseline
4. Hyperparameter tuning (GridSearchCV)
5. Sélection du meilleur modèle
6. Génération des visualisations

### Étape 3 : Évaluation finale sur test set

```bash
python evaluate_model.py
```

**Génère :**
- Métriques finales sur test set
- Matrice de confusion détaillée
- Courbe ROC
- Comparaison train/val/test
- Rapport d'évaluation complet

---

## 🔬 Méthodologie

### 1. Preprocessing

**⚠️ PRINCIPE CLÉ : Split AVANT preprocessing pour éviter le data leakage**

```
Dataset complet
      ↓
   SPLIT (stratifié)
      ↓
   ├─ Train (60%)
   ├─ Validation (20%)
   └─ Test (20%)
      ↓
Pipeline fitted sur TRAIN uniquement
      ↓
   ├─ Imputation (mode/médiane)
   ├─ Encodage ordinal + standardisation
   ├─ One-hot encoding (features nominales)
   └─ Standardisation (features numériques)
      ↓
Transformation de train/val/test
```

### 2. Gestion du déséquilibre

**Problème :** Dataset déséquilibré (~16% d'attrition)

**Solution :** `class_weight='balanced'` dans les modèles

### 3. Hyperparameter Tuning

- **Méthode :** GridSearchCV avec 3-fold cross-validation
- **Métrique d'optimisation :** ROC-AUC (adaptée aux classes déséquilibrées)
- **Modèles tunés :** Random Forest, XGBoost

### 4. Métriques d'évaluation

| Métrique | Description |
|----------|-------------|
| **Accuracy** | Taux de prédictions correctes |
| **Precision** | Parmi les prédictions "Left", combien sont correctes |
| **Recall** | Parmi les vrais "Left", combien sont détectés |
| **F1-Score** | Moyenne harmonique de Precision et Recall |
| **ROC-AUC** | Capacité à discriminer les classes (0.5 = random, 1.0 = parfait) |

**Métrique principale : ROC-AUC** (adaptée aux classes déséquilibrées)

---

## 📊 Résultats

### Comparaison des modèles (Validation Set)

| Modèle | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.XXX | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| Random Forest | 0.XXX | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| XGBoost | 0.XXX | 0.XXX | 0.XXX | 0.XXX | 0.XXX |

*(Les valeurs seront générées après exécution)*

### Meilleur modèle

**Modèle sélectionné :** [Sera déterminé après exécution]

**Performance sur Test Set :**
- ROC-AUC : X.XXX
- Accuracy : X.XXX
- F1-Score : X.XXX

### Top Features importantes

1. Feature 1 (importance: X.XXX)
2. Feature 2 (importance: X.XXX)
3. Feature 3 (importance: X.XXX)

---

## 🎯 Interprétation Business

### Facteurs d'attrition identifiés

Les features les plus importantes révèlent que l'attrition est principalement liée à :

1. **OverTime** : Les heures supplémentaires augmentent le risque de départ
2. **MonthlyIncome** : Les salaires bas sont corrélés à l'attrition
3. **YearsAtCompany** : Les nouveaux employés sont plus susceptibles de partir
4. **WorkLifeBalance** : Un mauvais équilibre augmente le turnover

### Recommandations RH

✅ **Actions préventives :**
- Limiter les heures supplémentaires
- Revoir les grilles salariales
- Programme d'intégration renforcé (0-2 ans)
- Améliorer la flexibilité et l'équilibre vie pro/perso

---

## ⚠️ Limitations

1. **Dataset limité** : Risque d'overfitting avec peu de données
2. **Données cross-sectionnelles** : Pas de validation temporelle
3. **Features potentiellement leaky** : MonthlyIncome pourrait être un proxy de la décision
4. **Classe minoritaire** : Difficulté à bien prédire les départs (16%)

---

## 🔮 Améliorations futures

- [ ] Tester d'autres algorithmes (LightGBM, CatBoost)
- [ ] Feature engineering avancé (interactions, polynômes)
- [ ] Analyse SHAP pour l'explicabilité
- [ ] Optimisation du seuil de classification
- [ ] Validation croisée stratifiée plus robuste
- [ ] Ensemble methods (voting, stacking)

---

## 👨‍💻 Auteur

**Souhaib MADHOUR**
- Module : Machine Learning
- Niveau : 4ème année Informatique et Réseaux
- Cycle d'ingénieur

---

## 📝 Notes techniques

### Éviter le data leakage

✅ **CORRECT :**
```python
# 1. Split AVANT preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 2. Fit preprocessing sur train uniquement
preprocessor.fit(X_train)

# 3. Transform train ET test
X_train_prep = preprocessor.transform(X_train)
X_test_prep = preprocessor.transform(X_test)
```

❌ **INCORRECT :**
```python
# Preprocessing AVANT split → DATA LEAKAGE!
X_preprocessed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y)
```

### Cross-validation

- **K-fold = 3** (compromis entre temps de calcul et robustesse)
- **Stratification** : Préserve la distribution des classes
- **Scoring = 'roc_auc'** : Métrique adaptée au déséquilibre

---

## 📚 Références

- Dataset : [IBM HR Analytics Employee Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- Scikit-learn Documentation : https://scikit-learn.org
- XGBoost Documentation : https://xgboost.readthedocs.io

---

**Bonne chance pour votre présentation ! 🚀**