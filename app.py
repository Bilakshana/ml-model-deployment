import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Student Performance Predictor")

st.title("🎓 Student Performance Prediction App")
st.write("This app predicts whether a student will PASS or FAIL based on input features.")

st.markdown("---")

study_hours = st.number_input("📚 Study Hours per Day", min_value=0.0, step=0.5)
attendance = st.number_input("📊 Attendance Percentage (%)", min_value=0.0, max_value=100.0)
previous_score = st.number_input("📝 Previous Exam Score", min_value=0.0, max_value=100.0)

st.markdown("---")


if st.button("🔍 Predict Result"):
    
    input_data = np.array([[study_hours, attendance, previous_score]])
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success("✅ The student is likely to PASS!")
    else:
        st.error("❌ The student is likely to FAIL.")

    st.write("Prediction Confidence:")
    st.write(f"Pass Probability: {probability[0][1]:.2f}")
    st.write(f"Fail Probability: {probability[0][0]:.2f}")

st.markdown("---")
st.caption("Developed by Bilakshana Neupane 💻✨")