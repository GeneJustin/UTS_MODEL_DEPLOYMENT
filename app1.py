import streamlit as st
import joblib
import numpy as np
import pandas as pd


logistic = joblib.load("artifacts/logistic.pkl")
linear = joblib.load("artifacts/linear.pkl")

def main():
    st.title("Placement & Salary Prediction")

    attendance = st.slider("Attendance Percentage", 0.0, 100.0, 75.0)
    study_hours = st.slider("Study Hours per Day", 0.0, 12.0, 4.0)
    backlogs = st.number_input("Backlogs", 0, 20, 0)
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)

    tenth = st.slider("10th Percentage", 0.0, 100.0, 75.0)
    twelfth = st.slider("12th Percentage", 0.0, 100.0, 75.0)
    projects = st.number_input("Projects Completed", 0, 20, 2)
    internships = st.number_input("Internships Completed", 0, 10, 0)
    coding = st.slider("Coding Skill Rating", 0, 10, 5)
    comm = st.slider("Communication Skill Rating", 0, 10, 5)
    aptitude = st.slider("Aptitude Skill Rating", 0, 10, 5)
    hackathons = st.number_input("Hackathons Participated", 0, 20, 0)
    certs = st.number_input("Certifications Count", 0, 20, 1)

    iob = st.selectbox("Internet Access", ["Yes", "No"])

    branch = st.selectbox("Branch", ["CSE", "IT", "ECE", "EEE", "MECH"])

    if st.button("Predict Placement & Salary"):
        input_dict = {
            "attendance_percentage": [attendance],
            "study_hours_per_day": [study_hours],
            "backlogs": [backlogs],
            "cgpa": [cgpa],
            "tenth_percentage": [tenth],
            "twelfth_percentage": [twelfth],
            "projects_completed": [projects],
            "internships_completed": [internships],
            "coding_skill_rating": [coding],
            "communication_skill_rating": [comm],
            "aptitude_skill_rating": [aptitude],
            "hackathons_participated": [hackathons],
            "certifications_count": [certs],
            "internet_access": [iob],
            "branch": [branch]
        }

        df = pd.DataFrame(input_dict)

        placement = logistic.predict(df)[0]
        salary = linear.predict(df)[0]

        st.subheader("Result")
        st.write("Placement Prediction:","Placed" if placement == 1 else "Not Placed")

        st.write(f"Predicted Salary: {salary:.2f}")


if __name__ == "__main__":
    main()