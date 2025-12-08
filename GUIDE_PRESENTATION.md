# 📊 Guide de Présentation - Projet HR Analytics

**Prédiction de l'Attrition des Employés**

---

## 👥 Répartition des Parties

| Personne | Partie | Durée | Difficulté |
|----------|--------|-------|------------|
| **Souhaib** | 1. Introduction & Dataset | 2-3 min | ⭐ Facile |
| **Kenza** | 2A. Preprocessing (Partie 1) | 3-4 min | ⭐⭐ Moyen |
| **Mohamed** | 2B. Preprocessing (Partie 2) | 3-4 min | ⭐⭐ Moyen |
| **Safia** | 3. Entraînement des Modèles | 5-6 min | ⭐⭐⭐ Un peu plus complexe |
| **Souhaib** | 4. Évaluation & Résultats | 6-7 min | ⭐⭐⭐⭐ Technique |
| **Tous** | 5. Conclusion & Questions | 2-3 min | - |

**Total : ~20-25 minutes**

---

# 🎯 PARTIE 1 : INTRODUCTION & DATASET
## 👤 Présenté par : **Souhaib**

### 📝 Ce que tu dois dire (2-3 minutes)

#### 1. Introduction du Projet (1 min)
```
"Bonjour, nous allons vous présenter notre projet de prédiction de l'attrition des employés.

L'objectif est de prédire si un employé va quitter l'entreprise ou non. 
C'est un problème de classification binaire en Machine Learning supervisé."
```

**Points clés à mentionner :**
- ✅ Classification binaire (Oui/Non)
- ✅ Apprentissage supervisé
- ✅ Utilité pour les RH : identifier les employés à risque

---

#### 2. Présentation du Dataset (2-3 min)

**A. Informations générales :**
```
"Notre dataset contient 1470 employés avec 35 caractéristiques (features).
Il s'agit du dataset IBM HR Analytics Employee Attrition."
```

**B. Distribution de la variable cible :**
- 📊 **Montrer le graphique** : `reports/target_distribution.png`
- 📈 **Dire les chiffres** :
  - **Stayed (Resté)** : 1233 employés (83.88%)
  - **Left (Parti)** : 237 employés (16.12%)
  - **Ratio** : 5.2:1 (déséquilibré !)

**C. Types de features :**
```
"Nous avons 3 types de features :
1. Numériques : Age, MonthlyIncome, YearsAtCompany...
2. Catégorielles ordinales : Education (1-5), JobSatisfaction (1-4)...
3. Catégorielles nominales : Department, Gender, JobRole..."
```

**D. Qualité des données :**
- ✅ Aucune valeur manquante
- ⚠️ Quelques valeurs aberrantes détectées (mais conservées)

---

### 🎨 Visualisations à montrer

1. **`reports/target_distribution.png`** - Distribution de la cible
2. **`reports/numerical_distributions.png`** - Distributions numériques (optionnel)
3. **`reports/correlations_with_attrition.png`** - Corrélations avec l'attrition

---

### 💡 Phrases de transition

**Fin de ta partie :**
```
"Maintenant, je vais vous expliquer la première partie du preprocessing : 
le split des données et la gestion des valeurs manquantes."
```

---

### ⚠️ Conseils pour Souhaib (Partie 1 + Preprocessing)

**Pour la partie Introduction & Dataset :**
- ✅ Reste simple et clair
- ✅ Pointe les graphiques avec la souris
- ✅ Parle lentement
- ❌ Ne rentre pas dans les détails techniques

**Pour la partie Preprocessing :**
- ✅ Tu connais ces concepts du cours (split, valeurs manquantes, outliers)
- ✅ Utilise des exemples concrets
- ✅ Insiste sur l'importance du split AVANT preprocessing
- ✅ Si tu oublies quelque chose, Souhaib peut compléter
- ✅ Parle avec confiance !

---

---

# 🔧 PARTIE 2A : PREPROCESSING (Partie 1)
## 👤 Présenté par : **KENZA**

### 📝 Ce que tu dois dire (3-4 minutes)

#### 1. Introduction au Preprocessing (30 sec)
```
"Le preprocessing est essentiel pour préparer les données avant l'entraînement.
Je vais vous présenter les premières étapes que nous avons étudiées en cours."
```

---

#### 2. Les Premières Étapes du Preprocessing (2.5-3.5 min)

**A. Split des données (1 min)**
```
"Première étape : nous avons divisé le dataset en 3 parties :
- Train (60%) : pour entraîner le modèle
- Validation (20%) : pour ajuster les hyperparamètres
- Test (20%) : pour évaluer le modèle final

⚠️ IMPORTANT : Le split se fait AVANT le preprocessing pour éviter le data leakage."
```

**📊 Montrer** : Diagramme du split (tu peux dessiner au tableau)

```
Dataset (1470)
    ↓
├─ Train (882) - 60%
├─ Validation (294) - 20%
└─ Test (294) - 20%
```

---

**B. Détection et imputation des valeurs manquantes (1 min)**
```
"Deuxième étape : gestion des valeurs manquantes.

✅ Résultat : Aucune valeur manquante dans notre dataset !
Mais si il y en avait, nous utiliserions :
- Mode (valeur la plus fréquente) pour les catégorielles
- Médiane pour les numériques

⚠️ L'imputation est calculée sur le TRAIN uniquement, puis appliquée à val/test."
```

---

**C. Traitement des valeurs aberrantes (1 min)**
```
"Troisième étape : détection des valeurs aberrantes.

Nous avons utilisé la méthode IQR (Interquartile Range) :
- Q1 - 1.5×IQR (borne inférieure)
- Q3 + 1.5×IQR (borne supérieure)

Top 5 features avec le plus d'outliers :
1. TrainingTimesLastYear : 127 outliers (14.40%)
2. MonthlyIncome : 67 outliers (7.60%)
3. YearsSinceLastPromotion : 59 outliers (6.69%)

⚠️ Nous avons CONSERVÉ les outliers car ils peuvent être informatifs pour la prédiction."
```

---

### 🎨 Visualisations à montrer

1. **`reports/preprocessing_summary.txt`** - Résumé du preprocessing
2. Diagramme du pipeline (dessiner au tableau)

---

### 💡 Phrases de transition

**Fin de ta partie :**
```
"Maintenant, Mohamed va continuer avec les étapes suivantes du preprocessing : 
l'encodage et la standardisation."
```

---

---

# 🔧 PARTIE 2B : PREPROCESSING (Partie 2)
## 👤 Présenté par : **MOHAMED**

### 📝 Ce que tu dois dire (3-4 minutes)

#### 1. Introduction (30 sec)
```
"Je vais continuer le preprocessing avec les étapes d'encodage et de standardisation."
```

---

#### 2. Les Étapes Suivantes du Preprocessing (2.5-3.5 min)

**A. Encodage (1.5-2 min)**
```
"Quatrième étape : encodage des features catégorielles.

1. Encodage ORDINAL pour les features avec ordre :
   - Education : 1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor
   - JobSatisfaction : 1=Low, 2=Medium, 3=High, 4=Very High
   - WorkLifeBalance : 1=Bad, 2=Good, 3=Better, 4=Best
   
   Pourquoi ordinal ? Parce que ces valeurs ont un ordre logique.

2. ONE-HOT ENCODING pour les features nominales (sans ordre) :
   - Department : Sales → [1,0,0], R&D → [0,1,0], HR → [0,0,1]
   - Gender : Male → [1,0], Female → [0,1]
   - BusinessTravel : Travel_Rarely → [1,0,0], Travel_Frequently → [0,1,0], Non-Travel → [0,0,1]
   
   Pourquoi one-hot ? Parce que ces valeurs n'ont pas d'ordre (ex: Sales ≠ R&D ≠ HR, mais aucun n'est "meilleur" qu'un autre).

Résultat : 35 features originales → ~50 features après encodage"
```

---

**B. Normalisation/Standardisation (1 min)**
```
"Cinquième étape : standardisation des features numériques.

Nous utilisons StandardScaler :
- Transforme les valeurs pour avoir moyenne=0 et écart-type=1
- Formule : (x - moyenne) / écart-type

Pourquoi ? Pour que toutes les features aient la même échelle.
Exemple : 
- MonthlyIncome : valeurs entre 0 et 20000
- Age : valeurs entre 18 et 60

Sans standardisation, MonthlyIncome dominerait Age car ses valeurs sont beaucoup plus grandes.

⚠️ IMPORTANT : Le scaler est FITTED sur TRAIN uniquement, puis transforme train/val/test.
Cela évite le data leakage."
```

---

#### 3. Pipeline de Preprocessing (30 sec)
```
"Toutes ces étapes sont combinées dans un pipeline sklearn :
- Pipeline fitted sur TRAIN uniquement
- Puis transformation de train, validation et test
- Sauvegardé pour réutilisation (preprocessing_pipeline.pkl)

Ce pipeline garantit que le preprocessing est fait de manière cohérente et reproductible."
```

**📊 Montrer** : `reports/preprocessing_summary.txt` (ouvrir rapidement)

---

### ⚠️ Conseils pour Mohamed

- ✅ Tu connais ces concepts du cours, tu peux t'appuyer dessus
- ✅ Utilise des exemples concrets (Education, Department...)
- ✅ Explique bien la différence entre ordinal et one-hot
- ✅ Insiste sur l'importance de fit sur TRAIN uniquement
- ✅ Si tu oublies quelque chose, Souhaib peut compléter
- ✅ Parle avec confiance, tu maîtrises cette partie !

---

---

# 🤖 PARTIE 3 : ENTRAÎNEMENT DES MODÈLES
## 👤 Présenté par : **SAFIA**

### 📝 Ce que tu dois dire (5-6 minutes)

#### 1. Introduction (1 min)
```
"Maintenant que les données sont prêtes, nous passons à l'entraînement des modèles.
Nous avons testé plusieurs algorithmes de classification supervisée."
```

---

#### 2. Modèles Testés (1 min)
```
"Nous avons testé 3 modèles baseline (sans tuning) :

1. Logistic Regression : modèle linéaire simple et interprétable
2. Random Forest : ensemble d'arbres de décision
3. XGBoost : gradient boosting, très performant

Tous avec class_weight='balanced' pour gérer le déséquilibre des classes."
```

---

#### 3. Comparaison Baseline (1-2 min)
```
"Après entraînement sur le train set, nous avons comparé les performances 
sur le validation set.

📊 Résultats (montrer le tableau ou graphique) :

| Modèle | ROC-AUC | Accuracy |
|--------|---------|----------|
| Logistic Regression | ~0.75 | ~0.85 |
| Random Forest | ~0.80 | ~0.84 |
| XGBoost | ~0.79 | ~0.83 |

✅ Conclusion : Random Forest et XGBoost ont les meilleures performances.
Nous avons donc décidé de faire le hyperparameter tuning uniquement sur ces 2 modèles."
```

**📊 Montrer** : 
- `reports/baseline_comparison.csv` (si disponible)
- `reports/roc_curves_comparison.png` - Comparaison des courbes ROC

---

#### 4. Hyperparameter Tuning (2-3 min)

**A. Pourquoi seulement 2 modèles ? (30 sec)**
```
"Pourquoi nous n'avons tuné que Random Forest et XGBoost ?

1. Efficacité computationnelle : le tuning prend beaucoup de temps
2. Performance : ce sont les 2 meilleurs modèles baseline
3. Meilleure pratique : optimiser les modèles les plus prometteurs"
```

---

**B. Méthode : GridSearchCV (1 min)**
```
"Nous avons utilisé GridSearchCV avec :
- Cross-validation : 5 folds (k=5)
- Métrique d'optimisation : ROC-AUC (adaptée au déséquilibre)
- Grid search : teste plusieurs combinaisons d'hyperparamètres

Pour Random Forest, nous avons testé :
- n_estimators : [100, 150]
- max_depth : [10, 15, 20]
- min_samples_split : [5, 10, 20]
- min_samples_leaf : [2, 4, 8]

Pour XGBoost :
- n_estimators : [100, 150]
- max_depth : [3, 5]
- learning_rate : [0.01, 0.1]
- subsample : [0.7, 0.8]
- Et d'autres paramètres de régularisation..."
```

---

**C. Résultats du Tuning (1 min)**
```
"Après le tuning, nous avons obtenu les meilleurs hyperparamètres pour chaque modèle.

Le modèle avec le meilleur ROC-AUC sur validation est sélectionné comme meilleur modèle.

Dans notre cas : Random Forest a été sélectionné comme meilleur modèle."
```

**📊 Montrer** :
- `reports/confusion_matrix_random_forest_tuned.png`
- `reports/confusion_matrix_xgboost_tuned.png`
- `reports/feature_importance_random_forest.png` - Top features importantes

---

#### 5. Gestion du Déséquilibre (30 sec)
```
"Rappel : notre dataset est déséquilibré (16% d'attrition).

Solutions appliquées :
- class_weight='balanced' : donne plus de poids à la classe minoritaire
- Métrique ROC-AUC : moins sensible au déséquilibre que l'accuracy"
```

---

### 🎨 Visualisations à montrer

1. **`reports/roc_curves_comparison.png`** - Comparaison des modèles
2. **`reports/confusion_matrix_random_forest_tuned.png`** - Matrice de confusion
3. **`reports/feature_importance_random_forest.png`** - Features importantes
4. **`reports/baseline_comparison.csv`** - Tableau de comparaison (si disponible)

---

### 💡 Phrases de transition

**Fin de ta partie :**
```
"Maintenant, Souhaib va vous présenter l'évaluation finale du modèle 
et les résultats obtenus."
```

---

### ⚠️ Conseils pour Safia

- ✅ Tu comprends mieux, donc tu peux expliquer plus en détail
- ✅ Utilise les graphiques pour illustrer
- ✅ Explique pourquoi on a choisi GridSearchCV et ROC-AUC
- ✅ Si question technique, Souhaib peut t'aider
- ✅ Sois confiante, tu maîtrises cette partie !

---

---

# 📈 PARTIE 4 : ÉVALUATION & RÉSULTATS
## 👤 Présenté par : **SOUHAIB**

### 📝 Ce que tu dois dire (6-7 minutes)

#### 1. Évaluation sur Test Set (2 min)
```
"Pour évaluer le modèle final, nous l'avons testé sur le test set 
qui n'a jamais été utilisé pendant l'entraînement."
```

**📊 Montrer** : `reports/final_evaluation_report.txt`

**Métriques obtenues :**
```
| Métrique | Train | Validation | Test |
|----------|-------|------------|------|
| Accuracy | 0.939 | 0.837 | 0.827 |
| Precision | 0.786 | 0.500 | 0.463 |
| Recall | 0.852 | 0.479 | 0.532 |
| F1-Score | 0.818 | 0.489 | 0.495 |
| ROC-AUC | 0.983 | 0.802 | 0.770 |
```

**Interprétation :**
- ✅ ROC-AUC = 0.77 sur test : Performance acceptable (>0.70)
- ⚠️ Écart train-test : 0.983 - 0.770 = 0.213 (signe d'overfitting)

---

#### 2. Analyse de l'Overfitting (2 min)
```
"Nous avons détecté un signe d'overfitting :
- Train ROC-AUC : 0.983 (très élevé)
- Test ROC-AUC : 0.770 (acceptable mais plus bas)

Écart : 0.213 (>0.10 = potentiel overfitting)

Causes possibles :
- Dataset relativement petit (1470 échantillons)
- Modèle trop complexe pour la taille des données
- Random Forest peut mémoriser les patterns du train

Solutions possibles :
- Augmenter la régularisation
- Réduire max_depth
- Augmenter min_samples_split/leaf
- Collecter plus de données"
```

**📊 Montrer** : 
- `reports/metrics_comparison.png` - Comparaison train/val/test
- `reports/final_test_result_overfitting.json` (optionnel)

---

#### 3. Matrice de Confusion (1 min)
```
"La matrice de confusion nous montre :
- True Negatives (TN) : Employés qui restent, prédits comme restant
- True Positives (TP) : Employés qui partent, prédits comme partant
- False Positives (FP) : Employés qui restent, prédits comme partant (faux alarmes)
- False Negatives (FN) : Employés qui partent, prédits comme restant (manqués)

Analyse :
- Le modèle détecte bien les employés qui restent (TN élevé)
- Plus de difficulté à détecter ceux qui partent (classe minoritaire)"
```

**📊 Montrer** : `reports/confusion_matrix_test_final.png`

---

#### 4. Courbe ROC (1 min)
```
"La courbe ROC montre la capacité du modèle à discriminer les classes :
- AUC = 0.77 : Bonne capacité de discrimination
- Meilleur que le hasard (0.5)
- Mais peut être amélioré"
```

**📊 Montrer** : `reports/roc_curve_test_final.png`

---

#### 5. Features Importantes (1 min)
```
"Les features les plus importantes pour la prédiction :
1. OverTime : Les heures supplémentaires sont un facteur clé
2. MonthlyIncome : Le salaire influence l'attrition
3. YearsAtCompany : Les nouveaux employés sont plus à risque
4. WorkLifeBalance : L'équilibre vie pro/perso est important
5. JobSatisfaction : La satisfaction au travail compte

Ces insights sont utiles pour les RH pour prendre des actions préventives."
```

**📊 Montrer** : `reports/feature_importance_random_forest.png`

---

#### 6. Interprétation Business (1 min)
```
"Recommandations pour les RH basées sur nos résultats :

✅ Actions préventives :
- Limiter les heures supplémentaires
- Revoir les grilles salariales
- Programme d'intégration renforcé (0-2 ans)
- Améliorer l'équilibre vie pro/perso
- Améliorer la satisfaction au travail"
```

---

### 🎨 Visualisations à montrer

1. **`reports/final_evaluation_report.txt`** - Rapport final
2. **`reports/metrics_comparison.png`** - Comparaison des métriques
3. **`reports/confusion_matrix_test_final.png`** - Matrice de confusion finale
4. **`reports/roc_curve_test_final.png`** - Courbe ROC finale
5. **`reports/feature_importance_random_forest.png`** - Features importantes

---

### 💡 Phrases de transition

**Fin de ta partie :**
```
"Pour conclure, nous allons faire un résumé du projet et répondre à vos questions."
```

---

### ⚠️ Conseils pour Souhaib

- ✅ Tu maîtrises tout, sois confiant
- ✅ Explique les métriques clairement
- ✅ Admet l'overfitting et propose des solutions
- ✅ Connecte les résultats techniques aux insights business
- ✅ Prépare-toi aux questions techniques

---

---

# 🎯 PARTIE 5 : CONCLUSION
## 👤 Présenté par : **TOUS**

### 📝 Ce que vous devez dire (2-3 minutes)

#### 1. Résumé du Projet (1 min)
```
"Pour résumer notre projet :

✅ Nous avons prédit l'attrition des employés avec un modèle Random Forest
✅ Performance : ROC-AUC = 0.77 sur test set (acceptable)
✅ Identifié les facteurs clés d'attrition (OverTime, Income, etc.)
✅ Pipeline complet et reproductible"
```

---

#### 2. Points Forts (30 sec)
```
"Points forts :
- Pipeline de preprocessing robuste (évite data leakage)
- Comparaison de plusieurs modèles
- Hyperparameter tuning avec GridSearchCV
- Évaluation complète avec métriques multiples"
```

---

#### 3. Limitations & Améliorations (1 min)
```
"Limitations :
- Dataset relativement petit (1470 échantillons)
- Signe d'overfitting détecté
- Classe minoritaire difficile à prédire (16%)

Améliorations futures :
- Collecter plus de données
- Tester d'autres algorithmes (LightGBM, CatBoost)
- Feature engineering avancé
- Analyse SHAP pour l'explicabilité"
```

---

#### 4. Remerciements (30 sec)
```
"Merci pour votre attention. Nous sommes prêts à répondre à vos questions."
```

---

---

# ❓ PRÉPARATION AUX QUESTIONS

## Questions Probables et Réponses

### Q1 : "Pourquoi seulement 2 modèles tunés ?"
**Réponse (Safia ou Souhaib)** :
"Nous avons testé 3 modèles baseline. Random Forest et XGBoost ont montré les meilleures performances. Le tuning est coûteux en temps, donc nous avons optimisé uniquement les plus prometteurs."

---

### Q2 : "Pourquoi ROC-AUC et pas Accuracy ?"
**Réponse (Souhaib)** :
"Notre dataset est déséquilibré (16% d'attrition). L'Accuracy peut être trompeuse (un modèle qui prédit toujours 'Stayed' aurait 84% d'accuracy). ROC-AUC est plus adaptée aux classes déséquilibrées."

---

### Q3 : "Comment avez-vous évité le data leakage ?"
**Réponse (Mohamed ou Souhaib)** :
"En faisant le split AVANT le preprocessing. Le pipeline est fitted uniquement sur le train set, puis transforme train, validation et test séparément."

---

### Q4 : "Pourquoi avez-vous gardé les outliers ?"
**Réponse (Mohamed)** :
"Les outliers peuvent être informatifs. Par exemple, un employé avec un salaire très élevé ou très bas peut être un facteur d'attrition. Nous les avons détectés mais conservés."

---

### Q5 : "Qu'est-ce que le class_weight='balanced' ?"
**Réponse (Safia ou Souhaib)** :
"C'est une technique pour gérer le déséquilibre. Le modèle donne plus de poids aux exemples de la classe minoritaire (attrition=Yes) pendant l'entraînement."

---

### Q6 : "Comment améliorer le modèle ?"
**Réponse (Souhaib)** :
"Plusieurs pistes : collecter plus de données, augmenter la régularisation, tester d'autres algorithmes, faire du feature engineering, ou utiliser des techniques d'ensemble."

---

---

# 📋 CHECKLIST AVANT LA PRÉSENTATION

## Pour TOUS

- [ ] Relire votre partie plusieurs fois
- [ ] Tester les visualisations (ouvrir les fichiers)
- [ ] Préparer des phrases de transition
- [ ] S'entraîner à parler lentement et clairement
- [ ] Prévoir des vêtements appropriés

## Pour KENZA

- [ ] Connaître les chiffres du dataset (1470, 83.88%, 16.12%)
- [ ] Savoir ouvrir les graphiques
- [ ] Comprendre le split des données (60/20/20)
- [ ] Connaître la gestion des valeurs manquantes et outliers
- [ ] Préparer les phrases de transition (vers preprocessing partie 2, puis vers Safia)

## Pour MOHAMED

- [ ] Comprendre l'encodage (ordinal vs one-hot)
- [ ] Connaître la standardisation (StandardScaler)
- [ ] Savoir expliquer pourquoi fit sur TRAIN uniquement
- [ ] Comprendre le pipeline sklearn
- [ ] Préparer la phrase de transition vers Safia

## Pour SAFIA

- [ ] Comprendre GridSearchCV et cross-validation
- [ ] Connaître les hyperparamètres testés
- [ ] Savoir expliquer pourquoi seulement 2 modèles
- [ ] Préparer la phrase de transition vers Souhaib

## Pour SOUHAIB

- [ ] Maîtriser toutes les métriques
- [ ] Comprendre l'overfitting et ses solutions
- [ ] Préparer les réponses aux questions techniques
- [ ] Connecter résultats techniques et business

---

# 🎤 CONSEILS GÉNÉRAUX

## Communication

- ✅ Parlez **lentement** et **clairement**
- ✅ **Regardez** le public (pas seulement l'écran)
- ✅ **Pointez** les graphiques avec la souris
- ✅ Utilisez des **gestes** pour appuyer vos propos
- ✅ **Souriez** et soyez confiants

## Technique

- ✅ Testez **avant** la présentation (ouvrir tous les fichiers)
- ✅ Ayez un **backup** (copie des fichiers sur clé USB)
- ✅ Préparez des **notes** (mais ne lisez pas directement)
- ✅ Anticipez les **questions** difficiles

## Gestion du Stress

- ✅ **Respirez** profondément avant de commencer
- ✅ Si vous oubliez quelque chose, **Souhaib peut compléter**
- ✅ Si question difficile, **dites "Je laisse Souhaib répondre"**
- ✅ **C'est normal** d'être un peu stressé, tout le monde l'est !

---

# 📁 FICHIERS À AVOIR PRÊTS

## Visualisations (dans `reports/`)

- [ ] `target_distribution.png`
- [ ] `correlations_with_attrition.png`
- [ ] `preprocessing_summary.txt`
- [ ] `roc_curves_comparison.png`
- [ ] `confusion_matrix_random_forest_tuned.png`
- [ ] `confusion_matrix_test_final.png`
- [ ] `roc_curve_test_final.png`
- [ ] `feature_importance_random_forest.png`
- [ ] `metrics_comparison.png`
- [ ] `final_evaluation_report.txt`

## Code (optionnel, pour questions techniques)

- [ ] `src/preprocessing.py`
- [ ] `src/train_models.py`
- [ ] `src/evaluate_model.py`

---

# 🎯 RÉSUMÉ RAPIDE PAR PERSONNE

## KENZA (5-7 min total)
**Partie 1 (2-3 min) :**
1. Introduction projet
2. Dataset : 1470 employés, 35 features
3. Distribution : 83.88% Stayed, 16.12% Left (déséquilibré)
4. Types de features
5. Qualité des données

**Partie 2A - Preprocessing (3-4 min) :**
1. Split : 60/20/20 (train/val/test) - AVANT preprocessing
2. Valeurs manquantes : aucune détectée
3. Outliers : détectés (méthode IQR) mais conservés

## MOHAMED (3-4 min)
**Partie 2B - Preprocessing (suite) :**
1. Encodage ordinal : Education, JobSatisfaction, etc. (avec ordre)
2. One-hot encoding : Department, Gender, etc. (sans ordre)
3. Standardisation : StandardScaler (moyenne=0, écart-type=1)
4. Pipeline sklearn : fitted sur TRAIN uniquement

## SAFIA (5-6 min)
1. 3 modèles baseline testés
2. Comparaison : RF et XGBoost meilleurs
3. Hyperparameter tuning : GridSearchCV (5-fold, ROC-AUC)
4. Sélection : Random Forest meilleur modèle
5. Gestion déséquilibre : class_weight='balanced'

## SOUHAIB (6-7 min)
1. Métriques test : ROC-AUC=0.77, Accuracy=0.83
2. Overfitting : écart train-test = 0.21
3. Matrice de confusion : analyse
4. Courbe ROC : AUC=0.77
5. Features importantes : OverTime, Income, etc.
6. Recommandations business

---

# 🚀 BONNE CHANCE ! 🍀

**Vous allez tous très bien présenter ! Restez calmes, parlez clairement, et n'hésitez pas à vous entraider.**

**Souhaib est là pour vous soutenir si besoin ! 💪**

---

**Dernière mise à jour : Décembre 2024**

