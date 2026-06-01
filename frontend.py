import streamlit as st
import requests 



API_URL = "http://127.0.0.1:8000/predict"
st.set_page_config(page_title="Diabetes Prediction ",layout= 'wide')
st.title("🩺 Diabetes Risk Prediction Model")
st.write("Enter the patient's details below to predict their diabetes risk .")
with st.form("prediction_form"):
    st.subheader("Demographics & Lifestyle")
    col1, col2, col3 = st.columns(3)
    # lhs is for the frontend value storage can be anything
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["male", "female", "other"])
        ethnicity = st.selectbox("Ethnicity", ["Hispanic", "White", "Asian", "Black", "Other"])
        ed_level = st.selectbox("Education Level", ["No formal", "Highschool", "Graduate", "Postgraduate"])
        
    with col2:
        income_lvl = st.selectbox("Income Level", ["Low", "Middle", "Lower-Middle", "Upper-Middle", "High"])
        emp_status = st.selectbox("Employment Status", ["Employed", "Retired", "Student", "Unemployed"])
        smoking_sts = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        
    with col3:
        alcohol_consumption_per_week = st.number_input("Alcohol (drinks/week)", min_value=0, value=1)
        physical_activity_minutes_per_week = st.number_input("Physical Activity (mins/week)", min_value=0, value=150)
        diet_score = st.number_input("Diet Score", min_value=0.0, max_value=10.0, value=5.0)
        sleep_hours_per_day = st.number_input("Sleep (hours/day)", min_value=0, max_value=24, value=8)
        screen_time_hours_per_day = st.number_input("Screen Time (hours/day)", min_value=0, max_value=24, value=4)

    st.markdown("---")
    st.subheader("Clinical Metrics & History")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.5)
        waist_to_hip_ratio = st.number_input("Waist-to-Hip Ratio", min_value=0.5, max_value=1.5, value=0.90)
        systolic_bp = st.number_input("Systolic BP", min_value=70, max_value=200, value=120)
        diastolic_bp = st.number_input("Diastolic BP", min_value=40, max_value=130, value=80)
        heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=75)
        
    with col5:
        cholesterol_total = st.number_input("Total Cholesterol", min_value=50, max_value=400, value=180)
        ldl_cholesterol = st.number_input("LDL Cholesterol", min_value=20, max_value=300, value=100)
        hdl_cholesterol = st.number_input("HDL Cholesterol", min_value=10, max_value=150, value=50)
        triglycerides = st.number_input("Triglycerides", min_value=20, max_value=500, value=150)
        
    with col6:
        st.write("Medical History")
        # Converting Yes/No to 1/0 for the API
        fam_hist = st.radio("Family History of Diabetes?", ["No", "Yes"])
        family_history_diabetes = 1 if fam_hist == "Yes" else 0
        
        hyp_hist = st.radio("History of Hypertension?", ["No", "Yes"])
        hypertension_history = 1 if hyp_hist == "Yes" else 0
        
        cardio_hist = st.radio("Cardiovascular History?", ["No", "Yes"])
        cardiovascular_history = 1 if cardio_hist == "Yes" else 0
    submitted = st.form_submit_button("Predict Diabetes Risk")
# lhs here should be same as in the INPUTUSER and rhs should be same as above used
    if submitted:
        payload={
        "age": age,
        "bmi": bmi,
        "income_lvl": income_lvl, # rhs is from the frontend and lhs from the backend
        "smoking_sts":smoking_sts,
        "ed_level": ed_level,
        "empl_status": emp_status,
        "cholesterol_total": cholesterol_total,
        "ldl_cholesterol": ldl_cholesterol,
        "triglycerides": triglycerides,
        "hdl_cholesterol": hdl_cholesterol,
        "waist_to_hip_ratio": waist_to_hip_ratio,
        "alcohol_consumption_per_week": alcohol_consumption_per_week,
        "physical_activity_minutes_per_week": physical_activity_minutes_per_week,
        "diet_score": diet_score,
        "sleep_hours_per_day": sleep_hours_per_day,
        "screen_time_hours_per_day": screen_time_hours_per_day,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "heart_rate": heart_rate,
        "gender": gender,
        "ethnicity": ethnicity,
        "family_history_diabetes": family_history_diabetes,
        "hypertension_history": hypertension_history,
        "cardiovascular_history": cardiovascular_history
        }
        with st.spinner("Analyzing patient data..."):
            try:
                # Send the data to your FastAPI server
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display the results
                    st.markdown("### 📊 Results")
                    probability = result['probability_score'] * 100
                    
                    if probability > 30.0:
                        st.error(f"**High Risk of Diabetes Detected.**")
                        st.write(f"The model predicts a **{probability:.2f}%** likelihood.")
                    else:
                        st.success(f"**Low Risk of Diabetes.**")
                        st.write(f"The model predicts a **{probability:.2f}%** likelihood.")
                else:
                    st.error(f"API Error: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API. Is your FastAPI server running?")

