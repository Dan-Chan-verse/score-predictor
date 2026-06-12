import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title(" Student Exam Score Predictor")
st.write("Adjust the student habits below to calculate a live score prediction.")

# 1. Load your newly trained model and columns blueprint
try:
    model = joblib.load('backbencher_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
except Exception as e:
    st.error(" Model files not found! Make sure you re-ran your notebook training cells.")
    st.stop()

# 2. Create User Interactive UI Inputs
study_hours = st.slider("Study Hours Per Day", 0.0, 12.0, 5.0)
attendance = st.slider("Attendance Percentage", 0.0, 100.0, 85.0)
sleep_hours = st.slider("Sleep Hours Per Night", 4.0, 12.0, 7.0)
mental_health = st.slider("Mental Health Rating (1-10)", 1, 10, 6)
part_time_job = st.selectbox("Does the student work a part-time job?", ["No", "Yes"])

st.write("---")

# 3. Build a Realistic Feature Vector Row matching the new model matrix
if st.button("Predict Exam Score"):
    # Create an empty row filled with zeros matching your new columns
    input_df = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
    
    # Fill in standard realistic averages for background variables
    if 'age' in input_df.columns: input_df['age'] = 20
    if 'social_media_hours' in input_df.columns: input_df['social_media_hours'] = 2.0
    if 'netflix_hours' in input_df.columns: input_df['netflix_hours'] = 1.5
    if 'exercise_frequency' in input_df.columns: input_df['exercise_frequency'] = 3.0
    
    # Default lifestyle options to moderate 'Good' settings
    if 'diet_quality_Good' in input_df.columns: input_df['diet_quality_Good'] = 1
    if 'internet_quality_Good' in input_df.columns: input_df['internet_quality_Good'] = 1
    
    # Map the custom interactive sliders dynamically
    if 'study_hours_per_day' in input_df.columns: 
        input_df['study_hours_per_day'] = study_hours
        
    if 'attendance_percentage' in input_df.columns: 
        input_df['attendance_percentage'] = attendance
        
    if 'sleep_hours' in input_df.columns: 
        input_df['sleep_hours'] = sleep_hours
        
    if 'mental_health_rating' in input_df.columns: 
        input_df['mental_health_rating'] = mental_health
        
    if 'part_time_job_Yes' in input_df.columns: 
        input_df['part_time_job_Yes'] = 1 if part_time_job == "Yes" else 0

    # 4. Compute the final output prediction
    prediction = model.predict(input_df)[0]
    final_score = max(0.0, min(100.0, prediction))
    
    st.success(f" Predicted Final Exam Score: **{final_score:.1f}%**")