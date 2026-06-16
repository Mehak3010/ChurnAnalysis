# 📊 Customer Churn Prediction using Machine Learning

## 📌 Project Overview

Customer churn significantly impacts profitability, customer acquisition costs, and long-term business growth. Retaining existing customers is often more cost-effective than acquiring new ones.

This project aims to identify customers likely to churn, understand the key factors influencing churn behavior, and develop predictive models to support proactive customer retention strategies.

---

## 🎯 Business Objectives

- Analyze customer behavior patterns associated with churn.
- Identify high-risk customer segments.
- Compare machine learning models for churn prediction.
- Translate analytical findings into actionable business recommendations.

---

## 📂 Dataset Information

**Dataset:** IBM Telco Customer Churn Dataset

**Records:** 7,032 customers

**Features:** 20 customer attributes including demographics, account information, billing details, and subscribed services.

**Target Variable:**

- **Yes** → Customer churned.
- **No** → Customer retained.

---

# 🔍 Exploratory Data Analysis (EDA)

## Customer Churn Distribution

The dataset exhibits class imbalance, with approximately **26.6% of customers churning**.

<img width="869" height="497" alt="image" src="https://github.com/user-attachments/assets/ba513ec0-b9f2-499e-a131-fbcf371ca900" />

---

## Churn Rate by Contract Type

Customers on **month-to-month contracts exhibit significantly higher churn rates** compared to those on longer-term contracts.

<img width="988" height="497" alt="image" src="https://github.com/user-attachments/assets/9dbb6245-9925-4d40-8453-40becb0a0573" />

---

## Churn Across Customer Lifecycle Stages

Customer churn is highest during the **first year of service**, indicating the importance of early customer engagement and onboarding initiatives.

<img width="867" height="497" alt="image" src="https://github.com/user-attachments/assets/3223ac6f-a361-44c0-9047-d1df44c23fac" />

---

## Churn Rate by Payment Method

Customers using **electronic checks demonstrate the highest churn propensity**, highlighting opportunities for intervention through payment modernization strategies.

<img width="1149" height="497" alt="image" src="https://github.com/user-attachments/assets/e8a44971-29d1-46ce-95a5-9cc4ad53e62e" />

---

# ⚙️ Feature Engineering

The following preprocessing steps were performed prior to model development:

- Removal of non-predictive identifiers (`customerID`)
- Target variable encoding
- One-hot encoding of categorical variables
- Stratified train-test split
- Standardization of numerical features

---

# 🤖 Machine Learning Models

The following classification algorithms were evaluated:

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier

Performance was assessed using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

# 📈 Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---------|----------|------------|---------|-----------|----------|
| Logistic Regression | 0.800 | 0.640 | 0.570 | 0.603 | 0.837 |
| Random Forest | 0.789 | 0.623 | 0.519 | 0.566 | 0.816 |
| XGBoost | 0.778 | 0.589 | 0.548 | 0.567 | 0.820 |

---

## Model Comparison Visualization

<img width="1028" height="574" alt="image" src="https://github.com/user-attachments/assets/71eb4c90-9cc2-4301-8c0c-4b6302194cc5" />

---

# 📉 ROC Curve Analysis

ROC curves were used to evaluate the discriminatory power of each model.

**Best Performing Model:** Logistic Regression

- Logistic Regression: **ROC-AUC = 0.837**
- Random Forest: **ROC-AUC = 0.816**
- XGBoost: **ROC-AUC = 0.820**

<img width="718" height="576" alt="image" src="https://github.com/user-attachments/assets/6a3d39ee-9e64-41ff-8405-a6260dc432c8" />

---

# 🔑 Key Drivers of Customer Churn

Feature importance analysis highlighted the most influential predictors associated with customer attrition.

Top predictors include:

- Contract Type
- Online Security
- Technical Support
- Internet Service Type
- Paperless Billing
- Phone Service
- 
<img width="1326" height="574" alt="image" src="https://github.com/user-attachments/assets/ebfc5f6c-9491-4779-bc88-94d93b87cd9c" />

---

# 💼 Business Recommendations

Based on the analytical findings and predictive modeling outcomes, the following recommendations are proposed:

### 1. Strengthen Early Customer Engagement

Prioritize retention initiatives during the first year of the customer lifecycle.

### 2. Promote Long-Term Contracts

Encourage migration from month-to-month plans through targeted incentives.

### 3. Improve Fiber Optic Customer Experience

Investigate service-related issues affecting fiber optic customers.

### 4. Encourage Adoption of Support Services

Promote Online Security and Technical Support offerings to enhance customer loyalty.

### 5. Modernize Payment Methods

Transition Electronic Check users toward automated payment alternatives.

### 6. Deploy Predictive Retention Programs

Leverage predictive models to identify high-risk customers and trigger personalized retention interventions.

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

---

# 📁 Repository Structure

```text
CustomerChurnAnalysis/
│
├── Customer_Churn_Analysis.ipynb
├── Telco-Customer-Churn.csv
├── README.md
├── requirements.txt
└── images/
    ├── churn_distribution.png
    ├── contract_type_churn.png
    ├── customer_lifecycle.png
    ├── payment_method_churn.png
    ├── model_comparison.png
    ├── roc_curve.png
    └── feature_importance.png
```

---

# 🚀 How to Run the Project

### Clone the repository

```bash
git clone https://github.com/Mehak3010/CustomerChurnAnalysis.git
```

### Navigate to the project directory

```bash
cd CustomerChurnAnalysis
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Customer_Churn_Analysis.ipynb
```

---

# 📌 Conclusion

This project demonstrates the application of machine learning techniques to predict customer churn and uncover the underlying factors influencing customer attrition.

The analysis revealed that contract type, tenure, payment behavior, service adoption, and internet service type play significant roles in customer retention outcomes.

Among the evaluated models, **Logistic Regression achieved the strongest overall performance**, providing both predictive effectiveness and interpretability.

The findings from this study may support organizations in developing proactive retention strategies aimed at reducing customer churn and improving long-term business value.

---

## 👩‍💻 Author

**Mehak Arora**

GitHub: https://github.com/Mehak3010

---
