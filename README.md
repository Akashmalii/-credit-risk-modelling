# 💳 Credit Risk Modelling

A machine learning project to assess and predict credit risk for loan applicants, classifying them into risk categories using statistical feature selection and ensemble models.

---

## 📌 Problem Statement

Financial institutions need to evaluate the creditworthiness of loan applicants to minimize default risk. This project builds a multi-class classification model that assigns applicants to one of four risk grades:

| Grade | Meaning |
|-------|---------|
| **P1** | Lowest risk — highly creditworthy |
| **P2** | Low risk |
| **P3** | Moderate risk |
| **P4** | Highest risk — likely to default |

---

## 📁 Project Structure

```
credit-risk-modelling/
│
├── Credit_Risk_Modelling.ipynb   # Main notebook with full pipeline
├── case_study1.xlsx              # Dataset 1 — applicant demographic data
├── case_study2.xlsx              # Dataset 2 — credit bureau / financial data
└── README.md
```

---

## 🔄 Workflow Overview

### 1. Data Loading & Merging
- Loaded two datasets (`case_study1.xlsx`, `case_study2.xlsx`)
- Merged on `PROSPECTID` using an inner join

### 2. Data Cleaning
- Removed rows and columns where sentinel value `-99999` indicated missing data
- Dropped columns with more than 10,000 missing entries

### 3. Feature Selection
- **Categorical features**: Chi-Square test against `Approved_Flag` (p-value ≤ 0.05)
  - Selected: `MARITALSTATUS`, `EDUCATION`, `GENDER`, `last_prod_enq2`, `first_prod_enq2`
- **Numerical features**: VIF (Variance Inflation Factor ≤ 6) to remove multicollinearity, followed by ANOVA (p-value ≤ 0.05)

### 4. Preprocessing Pipeline
- **Numerical**: Median imputation + Standard Scaling
- **Education**: Ordinal encoding with custom hierarchy (SSC → 12TH → GRADUATE → POST-GRADUATE)
- **Other Categorical**: Most-frequent imputation + One-Hot Encoding

### 5. Model Training & Comparison
Trained and compared 7 models:
- Logistic Regression
- Decision Tree
- Support Vector Classifier
- Naive Bayes
- Random Forest
- AdaBoost
- **XGBoost** ✅ *(best performer)*

### 6. Hyperparameter Tuning
XGBoost was tuned using `GridSearchCV` with 3-fold cross-validation.

**Best parameters:**
```
n_estimators   = 200
learning_rate  = 0.1
max_depth      = 5
gamma          = 0.2
min_child_weight = 3
```

### 7. Evaluation
- Accuracy Score
- Classification Report (Precision, Recall, F1-Score per class)
- Confusion Matrix heatmap

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| Pandas, NumPy | Data manipulation |
| Scikit-learn | Preprocessing, models, evaluation |
| XGBoost | Best-performing classifier |
| SciPy / Statsmodels | Chi-Square, ANOVA, VIF |
| Matplotlib, Seaborn | Visualizations |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-modelling.git
cd credit-risk-modelling
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost scipy statsmodels matplotlib seaborn openpyxl
```

### 3. Run the notebook
```bash
jupyter notebook Credit_Risk_Modelling.ipynb
```

> Make sure `case_study1.xlsx` and `case_study2.xlsx` are in the same directory as the notebook.

---

## 📊 Results

The final XGBoost model (after tuning) achieved the best accuracy among all tested models, with a detailed breakdown per risk grade visible in the classification report and confusion matrix.

---

## 🙋 Author

**Your Name**  
[GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)

---

## 📄 License

This project is for educational purposes. Feel free to use or adapt with attribution.
