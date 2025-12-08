# Reports Directory Documentation

This directory contains all the analysis reports, visualizations, and evaluation results generated during the Employee Attrition Prediction project.

## 📋 Table of Contents

1. [Text Reports](#text-reports)
2. [Visualization Files](#visualization-files)
3. [JSON Results](#json-results)
4. [Model Selection Strategy](#model-selection-strategy)

---

## 📄 Text Reports

### `eda_report.txt`
**Exploratory Data Analysis Report**

This report contains the initial analysis of the dataset:
- **Dataset dimensions**: 1470 rows × 35 columns
- **Target distribution**: 
  - No (Stayed): 1233 (83.88%)
  - Yes (Left): 237 (16.12%)
  - Ratio: 5.20:1 (highly imbalanced dataset)
- **Descriptive statistics** for numerical features
- **Missing values analysis**: No missing values detected
- **Recommendations** for model training (e.g., using `class_weight='balanced'`)

**Purpose**: Provides initial insights into the data structure, distribution, and quality before preprocessing.

---

### `preprocessing_summary.txt`
**Preprocessing Pipeline Summary**

This file documents the preprocessing steps applied to the data:
- **Dataset size**: 1470 rows, 35 columns
- **Missing values**: None detected
- **Outlier detection** (Top 5 features with most outliers):
  - TrainingTimesLastYear: 127 outliers (14.40%)
  - MonthlyIncome: 67 outliers (7.60%)
  - YearsSinceLastPromotion: 59 outliers (6.69%)
  - TotalWorkingYears: 37 outliers (4.20%)
  - NumCompaniesWorked: 31 outliers (3.51%)

**Preprocessing methodology**:
1. Stratified split (60% train, 20% validation, 20% test)
2. Missing value imputation (mode for categorical, median for numerical)
3. Ordinal encoding + standardization (for ordinal features)
4. One-hot encoding (for nominal features)
5. Standardization (for numerical features)
6. Pipeline fitted ONLY on training data (prevents data leakage)

**Purpose**: Documents the preprocessing pipeline to ensure reproducibility and prevent data leakage.

---

### `final_evaluation_report.txt`
**Final Model Evaluation Report**

Comprehensive evaluation of the best model on all datasets:
- **Model information**: Model name and training date
- **Performance metrics** across Train, Validation, and Test sets:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC
- **Overfitting analysis**: Detects potential overfitting by comparing train vs test performance
- **Performance assessment**: Indicates if the model meets acceptable thresholds (e.g., ROC-AUC ≥ 0.70)

**Purpose**: Provides a final assessment of model performance and generalization capability.

---

## 📊 Visualization Files

### Target Distribution
- **`target_distribution.png`**: Bar chart showing the distribution of the target variable (Attrition: Yes/No). Highlights the class imbalance in the dataset.

### Exploratory Data Analysis
- **`numerical_distributions.png`**: Distribution plots for numerical features (histograms, box plots, etc.)
- **`categorical_features_analysis.png`**: Analysis of categorical features showing their distributions and relationships with the target
- **`correlation_heatmap.png`**: Heatmap showing correlations between all numerical features
- **`correlations_with_attrition.png`**: Bar chart showing which features are most correlated with the target variable (Attrition)

### Model Evaluation Visualizations

#### Confusion Matrices
- **`confusion_matrix_logistic_regression.png`**: Confusion matrix for the baseline Logistic Regression model
- **`confusion_matrix_random_forest.png`**: Confusion matrix for the baseline Random Forest model
- **`confusion_matrix_random_forest_tuned.png`**: Confusion matrix for the tuned Random Forest model
- **`confusion_matrix_xgboost.png`**: Confusion matrix for the baseline XGBoost model
- **`confusion_matrix_xgboost_tuned.png`**: Confusion matrix for the tuned XGBoost model
- **`confusion_matrix_test_final.png`**: Final confusion matrix on the test set using the best model

**Purpose**: Visual representation of model predictions vs actual values, showing True Positives, True Negatives, False Positives, and False Negatives.

#### ROC Curves
- **`roc_curves_comparison.png`**: Comparison of ROC curves for all baseline models (Logistic Regression, Random Forest, XGBoost)
- **`roc_curve_test_final.png`**: Final ROC curve on the test set for the best model

**Purpose**: Shows the trade-off between True Positive Rate (TPR) and False Positive Rate (FPR) at different classification thresholds. Higher AUC indicates better model performance.

#### Feature Importance
- **`feature_importance_random_forest.png`**: Bar chart showing the top 15 most important features according to the Random Forest model
- **`feature_importance_xgboost.png`**: Bar chart showing the top 15 most important features according to the XGBoost model

**Purpose**: Identifies which features are most influential in making predictions, helping with model interpretability and feature selection.

#### Metrics Comparison
- **`metrics_comparison.png`**: Comparison of performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC) across Train, Validation, and Test sets

**Purpose**: Visualizes model performance across different datasets to detect overfitting and assess generalization.

---

## 📦 JSON Results

### `final_test_results.json`
**Final Test Set Evaluation Results**

Contains detailed metrics for the best model evaluated on all three datasets:
```json
{
  "model_name": "Random Forest",
  "train_metrics": {...},
  "validation_metrics": {...},
  "test_metrics": {...},
  "evaluation_date": "..."
}
```

**Metrics included**:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

**Purpose**: Machine-readable format for programmatic access to model performance metrics.

---

### `final_test_result_overfitting.json`
**Overfitting Analysis Results**

Similar structure to `final_test_results.json` but from a different evaluation run. Used to track model performance over time and detect overfitting patterns.

**Purpose**: Historical record of model evaluations for comparison and overfitting detection.

---

## 🎯 Model Selection Strategy

### Why Only XGBoost and Random Forest Were Tuned

The hyperparameter tuning process followed a **two-stage approach**:

#### Stage 1: Baseline Model Comparison
Three baseline models were trained without hyperparameter tuning:
1. **Logistic Regression** - Simple, interpretable linear model
2. **Random Forest** - Ensemble method with good generalization
3. **XGBoost** - Gradient boosting algorithm known for high performance

All models were evaluated on the validation set using ROC-AUC as the primary metric.

#### Stage 2: Selection for Hyperparameter Tuning
**Only the top 2 performing models** from the baseline comparison were selected for hyperparameter tuning:
- **Random Forest**
- **XGBoost**

**Reasons for this selection strategy**:

1. **Computational Efficiency**: 
   - Hyperparameter tuning with GridSearchCV is computationally expensive
   - Tuning all models would require significant time and resources
   - Focusing on the best performers maximizes return on investment

2. **Performance-Based Selection**:
   - The baseline comparison (`baseline_comparison.csv`) showed that Random Forest and XGBoost outperformed Logistic Regression
   - These two models demonstrated the highest ROC-AUC scores on the validation set
   - It's a common ML practice to tune only the most promising models

3. **Model Characteristics**:
   - **Random Forest**: Robust, handles non-linear relationships well, less prone to overfitting than single decision trees
   - **XGBoost**: State-of-the-art gradient boosting, excellent for structured/tabular data, often achieves top performance
   - **Logistic Regression**: Linear model with limited capacity; even with tuning, unlikely to match ensemble methods on complex patterns

4. **Resource Optimization**:
   - GridSearchCV with 5-fold cross-validation across multiple hyperparameter combinations is time-intensive
   - By focusing on 2 models instead of 3, we reduce tuning time by ~33% while maintaining high-quality results

5. **Best Practice**:
   - This approach follows the ML workflow: **explore → compare → optimize**
   - First explore multiple models quickly (baseline)
   - Then invest computational resources in optimizing the most promising candidates

#### Final Model Selection
After hyperparameter tuning, the model with the **highest ROC-AUC on the validation set** was selected as the final best model. In this case, it was **Random Forest**.

The selected model was saved as `models/best_model.pkl` and evaluated on the held-out test set to provide an unbiased estimate of generalization performance.

---

## 📁 File Organization

```
reports/
├── README.md                          # This file
├── eda_report.txt                     # Exploratory data analysis
├── preprocessing_summary.txt          # Preprocessing documentation
├── final_evaluation_report.txt        # Final model evaluation
├── final_test_results.json            # Test set metrics (JSON)
├── final_test_result_overfitting.json # Overfitting analysis (JSON)
├── target_distribution.png            # Target variable distribution
├── numerical_distributions.png        # Numerical features EDA
├── categorical_features_analysis.png  # Categorical features EDA
├── correlation_heatmap.png            # Feature correlations
├── correlations_with_attrition.png   # Target correlations
├── confusion_matrix_*.png             # Model confusion matrices
├── roc_curves_comparison.png          # ROC comparison
├── roc_curve_test_final.png           # Final ROC curve
├── feature_importance_*.png           # Feature importance plots
└── metrics_comparison.png             # Metrics comparison chart
```

---

## 🔍 How to Use These Reports

1. **Start with EDA**: Review `eda_report.txt` and visualization files to understand the data
2. **Check Preprocessing**: Review `preprocessing_summary.txt` to understand data transformations
3. **Evaluate Models**: Check confusion matrices and ROC curves to compare model performance
4. **Final Assessment**: Review `final_evaluation_report.txt` for the final model evaluation
5. **Metrics Access**: Use JSON files for programmatic access to metrics

---

## 📝 Notes

- All visualizations are saved in PNG format with 300 DPI for high quality
- Reports are generated automatically during the training and evaluation pipeline
- The reports directory is created automatically if it doesn't exist
- All paths in the code are relative to the project root directory

---

**Last Updated**: December 2024  
**Project**: HR Analytics - Employee Attrition Prediction  
**Author**: Souhaib MADHOUR

