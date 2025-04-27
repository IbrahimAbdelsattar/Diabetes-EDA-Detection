# Diabetes Prediction Project

This repository contains a machine learning model to predict diabetes based on various features from a dataset. The goal is to predict whether a patient has diabetes or not using a variety of health-related factors.

## Dataset Features

Below is a detailed description of each feature in the dataset:

### 🧓 1. Age
- **Description:** Age of the patient in years, ranging from 0.08 to 80.
- **Significance:** Risk of diabetes increases with age, particularly for Type 2 diabetes. The mean age is approximately 41.79 years.

### 🚻 2. Gender
- **Description:** Biological sex of the individual — Male, Female, or Other.
- **Significance:** Gender can influence diabetes susceptibility due to differences in hormones, body composition, and lifestyle factors. The dataset includes Male: 39,537 and Female: 55,262.

### ⚖️ 3. Body Mass Index (BMI)
- **Description:** A measure of body fat based on height and weight, with values in the dataset ranging from 10.16 to 71.55.
- **Categories:**
  - Underweight: < 18.5
  - Normal weight: 18.5 – 24.9
  - Overweight: 25 – 29.9
  - Obese: ≥ 30
- **Significance:** A higher BMI increases the likelihood of developing Type 2 diabetes. The most common value in the dataset is 27.32 (Overweight).

### ❤️ 4. Hypertension
- **Description:** Indicates if a patient has high blood pressure — 0 = No, 1 = Yes.
- **Distribution:** 
  - 0: 87,482
  - 1: 7,317
- **Significance:** Hypertension often coexists with diabetes and is linked to insulin resistance.

### 💔 5. Heart Disease
- **Description:** Indicates if the patient has any form of heart disease — 0 = No, 1 = Yes.
- **Distribution:** 
  - 0: 90,907
  - 1: 3,892
- **Significance:** Heart disease is both a complication and a risk factor for diabetes.

### 🚬 6. Smoking History
- **Description:** Smoking status of the patient.
- **Categories & Counts:**
  - Never: 34,011
  - No Info: 32,242
  - Former: 9,176
  - Current: 9,109
  - Not Current: 6,299
  - Ever: 3,962
- **Significance:** Smoking is known to contribute to insulin resistance and other metabolic issues that increase diabetes risk.

### 🧪 7. HbA1c Level
- **Description:** Reflects the average blood sugar level over the past 2–3 months.
- **Significance:** An HbA1c level of 6.5% or more typically indicates diabetes. Values in the dataset range from 3.5 to 9.0, with common values being 6.6, 5.7, 6.5, and 6.0.

### 💉 8. Blood Glucose Level
- **Description:** The current level of glucose in the blood.
- **Significance:** A crucial diagnostic marker — normal fasting blood glucose is generally between 70–130 mg/dL. Values above 250 are considered very high and potentially dangerous. The dataset contains values from 80 to 300, with common values like 159, 130, 126, and 140.

### 🎯 9. Diabetes
- **Description:** Target variable — 0 = No Diabetes, 1 = Has Diabetes.
- **Significance:** This is the outcome we're trying to predict based on the above features using machine learning models.

## Objective
The goal of this project is to create a machine learning model that can predict whether a person has diabetes based on the features provided in the dataset. The dataset includes information about various health metrics and demographic factors that may influence the likelihood of developing diabetes. 

By training machine learning algorithms on this dataset, we aim to predict the presence of diabetes and assist healthcare professionals in identifying high-risk individuals.
