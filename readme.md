# Employee Attrition Prediction

A machine learning project to predict employee attrition using HR analytics data. This project implements a complete ML pipeline including data preprocessing, feature engineering, model training, hyperparameter tuning, and model explainability.

## 📊 Project Overview

Employee attrition is a critical challenge for organizations. This project builds a predictive model to identify employees at risk of leaving, enabling proactive retention strategies.

**Key Features:**
- Comprehensive exploratory data analysis (EDA)
- Robust data preprocessing pipeline
- Multiple ML algorithms comparison (Logistic Regression, Random Forest, XGBoost)
- Handling class imbalance with SMOTE and class weights
- Hyperparameter tuning with cross-validation
- Model explainability using SHAP values
- Business-oriented threshold tuning

## 🗂️ Project Structure
```
employee-attrition-prediction/
├── data/                   # Dataset files (not tracked)
├── notebooks/              # Jupyter notebooks for EDA and experiments
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train_model.py
│   └── evaluate.py
├── models/                 # Trained models (not tracked)
├── reports/                # Generated reports and visualizations
│   └── figures/
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- pip

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/SouhaibMAD/employee-attrition-prediction.git
cd employee-attrition-prediction
```

2. **Create a virtual environment**
```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Download the dataset**
- Place your dataset in the `data/` folder
- Expected format: `data/employee_attrition.csv`
- Dataset link here : "https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data"

## 📈 Usage

### Running the Analysis

**Option 1: Jupyter Notebooks (Recommended for exploration)**
```bash
jupyter notebook
# Open notebooks in the notebooks/ folder sequentially
```

**Option 2: Python Scripts**
```bash
# Preprocess data
python src/preprocess.py

# Train model
python src/train_model.py

# Evaluate model
python src/evaluate.py
```

### Quick Start Example
```python
import pandas as pd
from src.preprocess import preprocess_data
from src.train_model import train_model

# Load data
df = pd.read_csv('data/employee_attrition.csv')

# Preprocess
X_train, X_test, y_train, y_test = preprocess_data(df)

# Train
model = train_model(X_train, y_train)

# Results saved in models/
```

## 🔍 Methodology

1. **Data Exploration**
   - Distribution analysis
   - Missing value assessment
   - Feature correlation analysis

2. **Data Preprocessing**
   - Missing value imputation
   - Feature encoding (one-hot, ordinal)
   - Feature scaling (StandardScaler)
   - Train/validation/test split (60/20/20)

3. **Feature Engineering**
   - Tenure binning
   - Income ratios
   - Interaction features

4. **Model Training**
   - Baseline: Dummy Classifier
   - Logistic Regression
   - Random Forest
   - XGBoost/LightGBM

5. **Handling Class Imbalance**
   - Class weights
   - SMOTE oversampling
   - Stratified sampling

6. **Evaluation Metrics**
   - Accuracy, Precision, Recall, F1-Score
   - ROC-AUC and Precision-Recall curves
   - Confusion Matrix

7. **Model Explainability**
   - Feature importance
   - SHAP values
   - Partial dependence plots

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Baseline | - | - | - | - | - |
| Logistic Regression | - | - | - | - | - |
| Random Forest | - | - | - | - | - |
| XGBoost | - | - | - | - | - |

*[Update with your actual results]*

### Key Findings
- **Top Predictive Features:** [e.g., Job Satisfaction, Monthly Income, Years at Company]
- **Model Performance:** [Brief summary]
- **Business Impact:** [Expected improvement in retention, cost savings, etc.]

## 🎯 Model Deployment (Optional)
```bash
# Run Streamlit demo app
streamlit run app.py
```

## 🛠️ Technologies Used

- **Python 3.11**
- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn, XGBoost, LightGBM, CatBoost
- **Imbalanced Learning:** imbalanced-learn (SMOTE)
- **Visualization:** matplotlib, seaborn
- **Explainability:** SHAP
- **Notebook:** Jupyter

## 📝 Project Status

- [x] Data exploration and preprocessing
- [x] Baseline model implementation
- [ ] Advanced feature engineering
- [ ] Hyperparameter tuning
- [ ] Model explainability analysis
- [ ] Final report and presentation
- [ ] Optional: Deployment demo

## 🤝 Contributing

This is an academic project. Feedback and suggestions are welcome!

## 📄 License

This project is created for educational purposes as part of a 4th-year ML course.

## 👤 Author

**Souhaib MADHOUR**
- GitHub: [@SouhaibMAD](https://github.com/SouhaibMAD)

## 🙏 Acknowledgments

- Dataset source : "https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data"
- Course instructor : Mme Ibtissam AMALOU

---

⭐ **If you find this project useful, please consider giving it a star!**