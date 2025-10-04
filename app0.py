import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page Config
st.set_page_config(page_title="Heart Attack Prediction App ❤️",
                   page_icon="❤️",
                   layout="wide")

st.title("🩺 Heart Attack Risk Prediction")
st.markdown("This app predicts the likelihood of a **heart attack** based on your health and lifestyle information.")

# Load Model + Preprocessor
model = joblib.load("mlp_diabetes_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")  # لازم تكون حافظه وانت بتدرب

# Sidebar info
st.sidebar.header("ℹ️ About")
st.sidebar.write("""
- Built with **Streamlit**  
- Model: **MLP Neural Network**  
- Data: CDC Heart Disease Indicators  
""")

# ============== User Inputs ==============
st.subheader("👤 Personal & Lifestyle Information")

col1, col2, col3 = st.columns(3)

with col1:
    state = st.selectbox("State", ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Unknown"])
    sex = st.selectbox("Sex", ["Male", "Female"])
    ageCategory = st.selectbox("Age Category", [
        "18-24", "25-29", "30-34", "35-39", "40-44",
        "45-49", "50-54", "55-59", "60-64", "65-69",
        "70-74", "75-79", "80+"
    ])
    raceEthnicity = st.selectbox("Race / Ethnicity", [
        "White", "Black", "Hispanic", "Asian", "American Indian", "Other", "Unknown"
    ])

with col2:
    generalHealth = st.selectbox("General Health", ["Excellent", "Very good", "Good", "Fair", "Poor"])
    lastCheckupTime = st.selectbox("Last Checkup Time", [
        "Within past year", "Within past 2 years", "Within past 5 years", "5+ years ago", "Never"
    ])
    physicalActivities = st.selectbox("Physical Activities", ["Yes", "No"])
    smokerStatus = st.selectbox("Smoker Status", ["Current", "Former", "Never", "Unknown"])
    ecigUsage = st.selectbox("E-Cigarette Usage", ["Every day", "Some days", "Not at all", "Unknown"])

with col3:
    chestScan = st.selectbox("Chest Scan", ["Yes", "No", "Unknown"])
    alcoholDrinkers = st.selectbox("Alcohol Drinkers", ["Yes", "No"])
    hivTesting = st.selectbox("HIV Testing", ["Yes", "No", "Unknown"])
    fluVaxLast12 = st.selectbox("Flu Vaccine (last 12 months)", ["Yes", "No"])
    highRiskLastYear = st.selectbox("High Risk Last Year", ["Yes", "No"])
    covidPositive = st.selectbox("COVID Positive", ["Yes", "No"])

# Medical conditions
st.subheader("🏥 Medical History")

col4, col5, col6 = st.columns(3)

with col4:
    heartAttack = st.selectbox("Had Heart Attack", ["Yes", "No"])
    angina = st.selectbox("Had Angina", ["Yes", "No"])
    stroke = st.selectbox("Had Stroke", ["Yes", "No"])
    asthma = st.selectbox("Had Asthma", ["Yes", "No"])

with col5:
    skinCancer = st.selectbox("Had Skin Cancer", ["Yes", "No"])
    copd = st.selectbox("Had COPD", ["Yes", "No"])
    depression = st.selectbox("Depression Disorder", ["Yes", "No"])
    kidneyDisease = st.selectbox("Kidney Disease", ["Yes", "No"])

with col6:
    arthritis = st.selectbox("Had Arthritis", ["Yes", "No"])
    diabetes = st.selectbox("Diabetes", ["Yes", "No"])
    hearingDifficulty = st.selectbox("Hearing Difficulty", ["Yes", "No"])
    visionDifficulty = st.selectbox("Vision Difficulty", ["Yes", "No"])

# Disabilities
st.subheader("♿ Disabilities & Daily Life")

col7, col8 = st.columns(2)

with col7:
    difficultyConcentrating = st.selectbox("Difficulty Concentrating", ["Yes", "No"])
    difficultyWalking = st.selectbox("Difficulty Walking", ["Yes", "No"])
    difficultyDressBath = st.selectbox("Difficulty Dressing / Bathing", ["Yes", "No"])

with col8:
    difficultyErrands = st.selectbox("Difficulty Running Errands", ["Yes", "No"])
    removedTeeth = st.selectbox("Removed Teeth", ["All", "Some", "None", "Unknown"])

# Numeric Features
st.subheader("📊 Health Measurements")

col9, col10, col11 = st.columns(3)

with col9:
    physicalHealthDays = st.number_input("Physical Health Days", min_value=0, max_value=30, value=0)
    mentalHealthDays = st.number_input("Mental Health Days", min_value=0, max_value=30, value=0)

with col10:
    sleepHours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
    heightM = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.70)

with col11:
    weightKg = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0)

# ============== Prediction Button ==============
if st.button("🔮 Predict"):
    # Put inputs into dataframe
    input_dict = {
        'State': state, 'Sex': sex, 'generalHealth': generalHealth,
        'physicalHealthDays': physicalHealthDays, 'mentalHealthDays': mentalHealthDays,
        'lastCheckupTime': lastCheckupTime, 'physicalActivities': physicalActivities,
        'sleepHours': sleepHours, 'removedTeeth': removedTeeth, 'heartAttack': heartAttack,
        'angina': angina, 'stroke': stroke, 'asthma': asthma, 'skinCancer': skinCancer,
        'copd': copd, 'depression': depression, 'kidneyDisease': kidneyDisease,
        'arthritis': arthritis, 'diabetes': diabetes, 'hearingDifficulty': hearingDifficulty,
        'visionDifficulty': visionDifficulty, 'difficultyConcentrating': difficultyConcentrating,
        'difficultyWalking': difficultyWalking, 'difficultyDressBath': difficultyDressBath,
        'difficultyErrands': difficultyErrands, 'smokerStatus': smokerStatus,
        'ecigUsage': ecigUsage, 'chestScan': chestScan, 'raceEthnicity': raceEthnicity,
        'ageCategory': ageCategory, 'heightM': heightM, 'weightKg': weightKg, 'bmi': bmi,
        'alcoholDrinkers': alcoholDrinkers, 'hivTesting': hivTesting,
        'fluVaxLast12': fluVaxLast12, 'highRiskLastYear': highRiskLastYear,
        'covidPositive': covidPositive
    }

    input_df = pd.DataFrame([input_dict])

    # Apply same preprocessing as training
    input_processed = preprocessor.transform(input_df)

    # Predict
    prediction = model.predict(input_processed)
    prob = model.predict_proba(input_processed)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Heart Attack (Probability: {prob:.2%})")
    else:
        st.success(f"✅ Low Risk of Heart Attack (Probability: {prob:.2%})")

