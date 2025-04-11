
# Telecom Customer Churn Prediction

## Project Overview

This project predicts telecom customer churn using a two-stage machine learning framework. We first predict Customer Lifetime Value (CLV) using regression and then use the predicted CLV with other features for churn classification. This approach enables churn prediction even without historical revenue data.

## Objectives

- Predict **Customer Lifetime Value (CLV)** using customer demographics and service usage data.
- Use **predicted CLV** as a feature to classify customers likely to churn.
- Interpret model outputs using **SHAP** to support explainable ML.
- Derive actionable insights for telecom providers to reduce churn.

---

## Dataset

We use the **Iranian Telecom Customer Churn dataset** from UCI:

- **Original shape**: 3150 rows × 14 features
- **Cleaned shape**: 2765 rows × 12 features (after removing duplicates, zero-usage customers, and `frequency_of_use`)
- Highly imbalanced (~15% churn rate)
- Includes features like: `call_failure`, `seconds_of_use`, `frequency_of_SMS`, `charge_amount`, `complains`, `tariff_plan`, `status`, and `age_group`.

📎 Source: [UCI Telecom Dataset](https://archive.ics.uci.edu/ml/datasets/iranian+churn+dataset)

---

## Methodology

### 🔧 Data Preparation

- Cleaned nulls, duplicates, and outliers.
- Removed 85 inactive zero-usage customer records.
- Engineered `customer_value` as the target variable for regression.
- Applied log transformation to `customer_value` to reduce skew.

### 📊 Exploratory Data Analysis

- Identified feature correlations and multicollinearity.
- Visualized churn trends across usage and demographic segments.

### 🔁 Two-Stage Modeling Framework

#### **Stage 1: CLV Prediction (Regression)**

- Models: Linear Regression, Polynomial Regression, Decision Tree, Random Forest
- Metric: RMSE, R² (with 5-fold CV)
- ✅ Best Model: **Random Forest** (R² = 0.995 on 5-fold cross-validation).


#### **Stage 2: Churn Classification**

- Models: Logistic Regression (with actual and predicted CLV), SVM (RBF), Random Forest, XGBoost
- Input:  All features, including predicted CLV.
- Metrics: Accuracy, F1-score, AUC
- ✅ Best Model: **XGBoost** (AUC = 0.9845)

---

## Model Interpretability

We used SHAP values (XGBoost) to understand feature contributions and model behavior.

- 🔑 Top drivers: `status_2`, `complains`, `frequency_of_SMS`, `seconds_of_use`, `customer_value`

---

## Insights & Recommendations

| Insight                                   | Recommendation                                          |
| ------------------------------------------ | ------------------------------------------------------- |
| `complains` strongly predicts churn      | Prioritize complaint handling strategies                |
| `status_2` = inactive → high churn risk  | Target re-engagement campaigns                          |
| High `frequency_of_SMS`, `seconds_of_use` → low churn | Encourage service usage and reward engagement                |
| `tariff_plan_2` linked to higher CLV     | Offer incentives to upgrade plans                       |
| Predicted CLV is effective                 | Use predicted CLV when revenue data isn't available |

---

## Project Structure
```
├── README.md
├── requirements.txt
├── venv/                         # Virtual environment
├── resources/
│   ├── data/
│   │   ├── input/                # Raw data
│   │   ├── output/               # Cleaned & processed
│   ├── models/                   # Saved model pipelines (.pkl files)
│       ├── random_forest_regression_model_final.pkl
│       ├── final_regression_pipeline.pkl
│       ├── best_churn_prediction_model.pkl
│       └── ... more models
├── notebooks/                    # Analysis & modeling notebooks
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_a_linear_regression.ipynb
│   ├── 03_b_linear_regression_model_intp.ipynb
│   ├── 04_a_logistic_regression.ipynb
│   ├── 04_b_logistic_regression_intp.ipynb
│   ├── 05_a_descision_reg.ipynb
│   ├── 06_a_svm_classifier.ipynb
│   ├── 07_a_rf_xgboost_classifier.ipynb
│   └── ... others
```
---

## How to Use This Project

```bash
# Clone the repository
git clone https://github.com/your-username/telecom-customer-churn-prediction.git
cd telecom-customer-churn-prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

---

## License

---

## Contact
👥 **AAI-501 Group 2**  
- Jagadeesh Kumar Sellappan  
- Jasmeet Kaur  
- Zoher Sabunwala

📫 For questions or contributions, feel free to open an issue or contact the team.

```