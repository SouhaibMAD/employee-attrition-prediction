## Project Setup & Reproducibility

### 1. **Create a Clean Project Folder:**
- Structure your project directory with the following folders:
  - `data/` – Raw data files
  - `notebooks/` – Jupyter notebooks for analysis
  - `src/` – Source code for preprocessing and modeling
  - `models/` – Trained models and pipelines
  - `reports/` – Project reports and slides

### 2. **Set Up a Virtual Environment:**
- Use `venv` or `conda` to create an isolated environment.
- Save dependencies using `requirements.txt` (for `pip`) or `environment.yml` (for `conda`).

### 3. **Set a Random Seed:**
- To ensure reproducibility, set a random seed at the top of your code:  
  ```python
  RANDOM_SEED = 42
  ```

### 4. **Version Your Dataset:**
- Name your dataset clearly and version it (e.g., `employee_attrition_v1.csv`) to ensure consistency in results.

---

## Data Loading

### 1. **Load the Dataset:**
- Import the data and verify it's loaded correctly:
  ```python
  import pandas as pd
  df = pd.read_csv('data/your_data.csv')
  ```

### 2. **Confirm Data:**
- Inspect the dataset's shape, preview the first few rows, and check column info:
  ```python
  df.shape
  df.head()
  df.info()
  ```

### Tips:
- Always check `.info()` and `.describe()` to understand the data types and basic stats.
- If the file is large, load a subset with `nrows=`.
  
### Pitfalls:
- Wrong encoding or corrupted CSV:  
  ```python
  pd.read_csv(..., encoding='utf-8', errors='replace')
  ```

---

## Basic Exploratory Data Analysis (EDA)

### Goal: Understand Distributions, Relationships, and the Target Variable

### 1. **Target Distribution:**
- Get the distribution of the target variable:
  ```python
  df['target'].value_counts(normalize=True)
  ```

### 2. **Missing Values:**
- Check for missing values:
  ```python
  df.isna().sum()
  ```

### 3. **Visualize Numeric Features:**
- Plot histograms and boxplots for numeric features:
  ```python
  df['Age'].hist()
  df.boxplot(column='MonthlyIncome', by='Attrition')
  ```

### 4. **Categorical Feature Counts:**
- Plot bar charts for categorical variables:
  ```python
  df['JobRole'].value_counts().plot(kind='bar')
  ```

### 5. **Pairwise Correlations:**
- Check correlations between numeric features:
  ```python
  df.corr()
  ```

### Tips:
- Visuals help to spot outliers and skewed distributions (use log transformation for skewed data).
- Explore relationships using group-by:  
  ```python
  df.groupby('JobRole')['target'].mean()
  ```

### Pitfalls:
- Don't rely solely on correlations, especially with categorical variables.

---

## Data Cleaning

### Goal: Tidy Up the Dataset for Modeling

### 1. **Handle Missing Values:**
- Drop rows with very few missing values or impute missing values:
  - Numeric: impute with median
  - Categorical: impute with mode

### 2. **Fix Data Types:**
- Convert boolean and categorical columns to the appropriate `category` dtype.

### 3. **Remove Duplicates:**
  ```python
  df.drop_duplicates(inplace=True)
  ```

### 4. **Correct Obvious Errors:**
- Ensure values like age are within reasonable ranges.

### Tips:
- Keep a log of cleaning steps and describe changes in code comments.
- Mark imputed values with a binary flag.

### Pitfalls:
- Avoid leaking future information into features during data cleaning.

---

## Feature Engineering (Minimal, High-Impact)

### Goal: Create a Few Useful Features

### 1. **Simple Transforms:**
- **Binning**: Convert continuous variables into categories (e.g., tenure into buckets).
- **Ratios**: Create new features based on ratios (e.g., `IncomePerYear = MonthlyIncome * 12`).
- **Interaction**: Multiply features that might interact (e.g., `OverTime * JobSatisfaction`).

### 2. **Encoding:**
- One-hot encode small-cardinality categorical features.
- Map ordinal features to integers.

### 3. **Scaling:**
- Use `StandardScaler` for linear models (e.g., Logistic Regression).
- Tree-based models (e.g., Random Forest) don’t need scaling.

### Tips:
- Start small: Create 3–6 engineered features that make domain sense.
- Use a reproducible pipeline (e.g., `ColumnTransformer`) to save preprocessing steps.

### Pitfalls:
- Avoid creating too many sparse one-hot columns for high-cardinality features.

---

## Train / Validation / Test Split

### Goal: Reliable Evaluation Without Leakage

### 1. **Stratified Split:**
- Ensure the target distribution is maintained across splits:
  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_SEED)
  X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_SEED)
  ```

### Tips:
- Keep the test set untouched until final evaluation.
- For small datasets, use cross-validation (e.g., `StratifiedKFold`).

### Pitfalls:
- Perform feature selection only after splitting the data.

---

## Baseline Model & Simple Checks

### Goal: Establish a Baseline to Beat

### 1. **Create a Baseline Model:**
- A simple model to set expectations, such as a Dummy Classifier predicting the majority class or a Logistic Regression with a few features:
  ```python
  from sklearn.dummy import DummyClassifier
  baseline = DummyClassifier(strategy='most_frequent')
  baseline.fit(X_train, y_train)
  ```

### Tips:
- A baseline helps set expectations — document it in your report.

### Pitfalls:
- Avoid starting with complex models before understanding baseline performance.

---

## Model Selection & Training

### Goal: Try Multiple Models and Pick the Best

### 1. **Model Candidates:**
- Start with:
  - Logistic Regression (interpretable)
  - Decision Tree (interpretable, visual)
  - Random Forest (robust)
  - Gradient Boosting (e.g., XGBoost, LightGBM) if comfortable

### 2. **Pipelines:**
- Use scikit-learn pipelines for combining preprocessing and modeling:
  ```python
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler, OneHotEncoder
  from sklearn.compose import ColumnTransformer

  preproc = ColumnTransformer([('num', StandardScaler(), num_cols), ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)])
  pipe = Pipeline([('preproc', preproc), ('clf', LogisticRegression(max_iter=1000))])
  pipe.fit(X_train, y_train)
  ```

### Tips:
- Use `class_weight='balanced'` for models like Logistic Regression and Decision Trees if the target is imbalanced.
- Fit only on the training set and evaluate on the validation set.

### Pitfalls:
- Watch for overfitting: monitor train vs. validation performance.

---

## Handling Class Imbalance

### Goal: Ensure Minority Class is Learned

### 1. **Techniques:**
- Use class weights (e.g., for Logistic Regression).
- Apply oversampling (e.g., SMOTE) or undersampling.

### 2. **Pipeline Integration:**
- Apply resampling only to the training data within a pipeline.

### Tips:
- Evaluate using precision/recall, F1 score, and PR curve.

### Pitfalls:
- Avoid applying SMOTE before splitting the data, as it can lead to leakage.

---

## Hyperparameter Tuning & Cross-Validation

### Goal: Refine Models Using Cross-Validation

### 1. **Search for Optimal Parameters:**
- Use `RandomizedSearchCV` or `GridSearchCV`:
  ```python
  from sklearn.model_selection import RandomizedSearchCV
  search = RandomizedSearchCV(pipe, param_distributions=..., scoring='f1', cv=5)
  search.fit(X_train, y_train)
  ```

### Tips:
- Limit the search space to avoid running out of time.
- Use `n_iter` to control runtime.

### Pitfalls:
- Don’t tune on the test set — reserve it for final evaluation.

---

## Evaluation & Metrics

### Goal: Measure Model Quality

### 1. **Key Metrics:**
- Confusion matrix: True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
- Precision, Recall, F1-score
- ROC AUC and Precision-Recall curve

### 2. **Evaluation Code:**
  ```python
  from sklearn.metrics import classification_report, roc_auc_score
  print(classification_report(y_test, y_pred))
  ```

### Tips:
- For imbalanced datasets, prioritize PR curve and recall over accuracy.
