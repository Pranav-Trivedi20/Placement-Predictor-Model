import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
model = joblib.load(os.path.join(os.path.dirname(__file__), "placement_logistic_model.pkl"))
scaler = joblib.load(os.path.join(os.path.dirname(__file__), "scaler.pkl"))


data_path = os.path.join(os.path.dirname(__file__), "college_student_placement_dataset.csv")
df = pd.read_csv(data_path)
df['Internship_Experience'] = df['Internship_Experience'].map({'Yes': 1, 'No': 0})
df['Placement'] = df['Placement'].map({'Yes': 1, 'No': 0})
df['Internship_Experience'] *= 10  

placed_data = df[df['Placement'] == 1]
avg_scores = placed_data.select_dtypes(include=[np.number]).mean()


st.title("🎓 College Placement Predictor")
st.warning(
    "⚠️ This is a predictive tool for learning purposes. Actual placement depends on multiple real-life factors."
)

with st.form("placement_form"):
    iq = st.number_input("IQ Score", min_value=50, max_value=200, value=100)
    prev_sem = st.number_input("Previous Semester GPA", min_value=0.0, max_value=10.0, value=7.5)
    cgpa = st.number_input("Cumulative GPA (CGPA)", min_value=0.0, max_value=10.0, value=7.5)
    academic = st.slider("Academic Performance (1-10)", 1, 10, 5)
    internship = st.number_input("Internship Experience (Number of Internships)", min_value=0, max_value=10, value=0)
    extracurricular = st.slider("Extra-Curricular Score (0-10)", 0, 10, 5)
    communication = st.slider("Communication Skills (1-10)", 1, 10, 5)
    projects = st.number_input("Projects Completed (Major Projects)", min_value=0, max_value=20, value=1)
    
    submitted = st.form_submit_button("Predict Placement")

if submitted:
    user_data = pd.DataFrame([{
        'IQ': iq,
        'Prev_Sem_Result': prev_sem,
        'CGPA': cgpa,
        'Academic_Performance': academic,
        'Internship_Experience': internship * 10,  
        'Extra_Curricular_Score': extracurricular,
        'Communication_Skills': communication,
        'Projects_Completed': projects
    }])

    scaled_data = scaler.transform(user_data)
    prediction = model.predict(scaled_data)[0]

    if prediction == 1:
        st.success("✅ Congratulations! You have high chances of getting placed.")
    else:
        st.error("❌ Prediction: Not Placed")

        st.subheader("📊 Suggestions to Improve Chances:")

        suggestions = []
        features = [
            'IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance',
            'Internship_Experience', 'Extra_Curricular_Score',
            'Communication_Skills', 'Projects_Completed'
        ]

        for col in features:
            your_score = user_data[col].iloc[0]
            avg_score = avg_scores[col] / (10 if col == 'Internship_Experience' else 1)
            if col in ['Internship_Experience', 'Projects_Completed']:
                your_score_display = int(your_score) if col == 'Projects_Completed' else int(your_score / 10)
                avg_score_display = int(round(avg_score))
            else:
                your_score_display = round(your_score, 2)
                avg_score_display = round(avg_score, 2)

            if your_score < avg_score:
                suggestions.append(
                    f"- **{col.replace('_', ' ')}**: Your score ({your_score_display}) "
                    f"is below the average of placed students ({avg_score_display})."
                )

        if suggestions:
            for s in suggestions:
                st.markdown(s)
        else:
            st.info("You are close to the average scores of placed students. Keep improving!")

