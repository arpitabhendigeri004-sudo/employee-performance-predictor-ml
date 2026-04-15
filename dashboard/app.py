import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# -------------------------------
# LOAD MODEL
# -------------------------------
base_dir = os.path.dirname(__file__)

model = joblib.load(os.path.join(base_dir, "../models/model.pkl"))
train_cols = joblib.load(os.path.join(base_dir, "../models/columns.pkl"))

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Employee Performance AI", layout="wide")

st.title("🚀 Employee Performance Predictor (AI Powered)")

st.sidebar.header("Employee Inputs")

age = st.sidebar.slider("Age", 20, 60, 30)
experience = st.sidebar.slider("Experience", 1, 30, 5)
salary = st.sidebar.number_input("Salary", 20000, 150000, 50000)
projects = st.sidebar.slider("Projects", 1, 15, 5)
training = st.sidebar.slider("Training Hours", 0, 100, 20)
feedback = st.sidebar.slider("Feedback Score", 1.0, 5.0, 3.0)
department = st.sidebar.selectbox("Department", ["IT", "HR", "Sales"])

# -------------------------------
# INPUT PREP
# -------------------------------
input_df = pd.DataFrame([{
    'age': age,
    'experience': experience,
    'salary': salary,
    'training_hours': training,
    'projects': projects,
    'feedback_score': feedback,
    'department': department
}])

input_df = pd.get_dummies(input_df)

for col in train_cols:
    if col not in input_df:
        input_df[col] = 0

input_df = input_df[train_cols]

# -------------------------------
# BUTTON (ALL LOGIC INSIDE)
# -------------------------------
if st.sidebar.button("Predict Performance", key="predict_btn"):

    prediction = model.predict(input_df)[0]

    # RESULT
    st.markdown("### 🎯 Prediction Result")
    if prediction == 1:
        st.error("⚠️ Low Performer")
    else:
        st.success("🌟 High / Medium Performer")

    st.divider()

    # FEATURE IMPORTANCE
    st.markdown("### 📊 Feature Importance")

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": train_cols,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(8)

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # HR RECOMMENDATIONS
    st.markdown("### 💡 HR Insights & Recommendations")

    recommendations = []

    if projects < 4:
        recommendations.append("📌 Increase project involvement")

    if feedback < 3:
        recommendations.append("📌 Improve feedback via mentorship")

    if training < 20:
        recommendations.append("📌 Assign skill development programs")

    if experience < 3:
        recommendations.append("📌 Provide senior mentorship")

    if len(recommendations) == 0:
        st.success("✅ Employee is performing well with no major risks detected.")
    else:
        for rec in recommendations:
            st.warning(rec)