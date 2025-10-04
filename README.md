# 🫀 Heart Attack Risk Prediction

> Predicting individual risk of heart attack using CDC BRFSS health indicators and a trained MLP model.  
> This repository contains the data processing, model training, evaluation, and a Streamlit app for easy deployment.

---

## 🚀 Project Overview

Cardiovascular disease (including heart attacks) is one of the leading causes of death worldwide. Early identification of individuals at high risk enables targeted prevention, timely care, and better allocation of healthcare resources.

This project uses behavioral and health indicators from the BRFSS survey to build a machine learning model that predicts whether an individual has experienced a heart attack (`heartAttack`). 
Kaggle Dataset (https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/versions/6)
The final deliverable is a trained **MLP (Multi-Layer Perceptron) model** and a polished **Streamlit web app** that accepts user inputs and returns an individualized risk prediction with probability.

---

## ✅ Why This Project Matters

- **Public health impact** → Helps identify high-risk individuals for early intervention.  
- **Actionable insights** → Detects important risk factors (BMI, smoking, chronic conditions, activity).  
- **Accessible deployment** → Streamlit app allows non-technical users (clinicians, public health workers, individuals) to obtain quick risk estimates.  
- **Reproducible pipeline** → End-to-end workflow from preprocessing → modeling → evaluation → deployment.  

---

## 📁 Repository Structure
├─ data/ # raw and processed data (not committed for privacy)
├─ notebooks/ # EDA and experiments
├─ src/
│ ├─ data_preprocessing.py # preprocessing, encoders, scalers
│ ├─ train.py # training pipeline (train/val/test, SMOTE, fit models)
│ ├─ evaluate.py # evaluation utilities, plots
│ └─ utils.py # helper functions
├─ models/
│ ├─ mlp_model.pkl # trained MLP model (joblib)
│ ├─ preprocessor.pkl # preprocessing pipeline
│ └─ model_features.pkl # ordered feature names
├─ app/
│ └─ streamlit_app.py # Streamlit UI for deployment
├─ requirements.txt
└─ README.md

---

---

Do you want me to also **generate a shorter version** (like a summary style README) for people who just want to quickly understand and run the app, or you prefer to keep this full detailed one?

## 🧭 Data

- **Source**: BRFSS (Behavioral Risk Factor Surveillance System — CDC)  
- **Target**: `heartAttack` (Yes/No)  
- **Features**: demographic, lifestyle, and health-related indicators  
  (examples: `generalHealth`, `smokerStatus`, `bmi`, `diabetes`, `stroke`, `ageCategory`, etc.)

⚠️ **Note**: Do not commit or share personally identifying information (PII).  

---

## 🛠️ Methodology

1. **Exploratory Data Analysis (EDA)**  
   - Explore distributions, missing values, and class imbalance  
   - Visualize risk factors (BMI vs heartAttack, ageCategory, etc.)  

2. **Preprocessing**  
   - Handle missing values (`Unknown` categories / median imputation)  
   - Normalize numerical features (`StandardScaler`)  
   - Encode categorical features (Label/Ordinal/OneHot)  
   - Save preprocessing pipeline (`preprocessor.pkl`)  

3. **Train / Validation / Test Split**  
   - Stratified split (60% train, 20% validation, 20% test)  
   - Apply **SMOTE** on training set only to handle imbalance  

4. **Modeling**  
   - Baseline: Logistic Regression, Random Forest  
   - Final: **MLP Classifier (neural network)** with early stopping  

5. **Evaluation**  
   - Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC  
   - SHAP analysis for feature importance  

6. **Model Persistence**  
   - Save trained model (`mlp_model.pkl`) and pipeline (`preprocessor.pkl`)  

7. **Deployment**  
   - **Streamlit app** accepts user inputs and returns prediction + probability  
   - Can be deployed to **Streamlit Cloud, Heroku, AWS, or Docker**  

---

## 🧪 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
