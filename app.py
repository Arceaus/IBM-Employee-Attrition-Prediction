import joblib
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
import plotly.express as px

from config import *

st.set_page_config(
    page_title="IBM Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
background_data = joblib.load(BACKGROUND_DATA_PATH)

explainer = shap.LinearExplainer(model,background_data)

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


st.title("📊 IBM Employee Attrition Prediction")

st.markdown("""
Predict employee attrition using a Machine Learning model trained on the IBM HR Analytics dataset.

Use the sidebar to enter employee information or upload a CSV for bulk predictions.
""")

st.divider()

st.sidebar.header("Employee Information")

st.sidebar.markdown("---")

st.sidebar.subheader("👤 **Personal**")

age = st.sidebar.number_input("**Enter Age**",min_value=18,max_value=60,value=30)

gender = st.sidebar.selectbox("**Gender**",["Male","Female"])

marital_status = st.sidebar.selectbox("**Marital Status**",["Single","Divorced","Married"])

st.sidebar.markdown("---")

st.sidebar.subheader("💼 **Job**")

department = st.sidebar.selectbox("**Department**",["Sales","Human Resources","Research & Development"])

job_role = st.sidebar.selectbox("**Job Role**",['Sales Executive', 'Research Scientist', 'Laboratory Technician',
       'Manufacturing Director', 'Healthcare Representative', 'Manager',
       'Sales Representative', 'Research Director', 'Human Resources'])

job_level = st.sidebar.selectbox(
    "**Job Level**",
    [
        "1 - Entry Level",
        "2 - Junior",
        "3 - Mid Level",
        "4 - Senior",
        "5 - Executive"
    ]
)

job_level = int(job_level[0])

overtime = st.sidebar.selectbox("**Overtime**",["Yes","No"])

st.sidebar.markdown("---")

st.sidebar.subheader("🎓 **Education**")

education = st.sidebar.selectbox("**Education**",["1 - Below College","2 - College","3 - Bachelor", "4 - Master","5 - Doctorate/PHD"])
education = int(education[0])

education_field = st.sidebar.selectbox("**Education Field**",['Life Sciences', 'Medical', 'Marketing','Technical Degree', 'Human Resources','Other'])

st.sidebar.markdown("---")

st.sidebar.subheader("💰 **Salary**")

monthly_income = st.sidebar.number_input("**Salary**",min_value=1000,max_value=50000,value=15000,step=500)

stock_opt_level = st.sidebar.selectbox("**Stock Option Level**",[0,1,2,3],help="Stock option level provided in the IBM HR dataset(0-3)")

percent_salary_hike = st.sidebar.selectbox("**Percent Salary Hike (%)**",[11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])

st.sidebar.markdown("---")

st.sidebar.subheader("💰 **Salary Metrics**")

hourly_rate = st.sidebar.number_input(
    "**Hourly Rate**",
    min_value=30,
    max_value=100,
    value=65
)

daily_rate = st.sidebar.number_input(
    "**Daily Rate**",
    min_value=100,
    max_value=1500,
    value=800
)

monthly_rate = st.sidebar.number_input(
    "**Monthly Rate**",
    min_value=2000,
    max_value=30000,
    value=14000
)

st.sidebar.markdown("---")

st.sidebar.subheader("❤️ **Satisfaction**")

environment_satisfaction = st.sidebar.selectbox("**Environment Satisfaction**",["1 - Low","2 - Medium","3 - High","4 - Very High"])
environment_satisfaction = int(environment_satisfaction[0])

job_satisfaction = st.sidebar.selectbox("**Job Satisfaction**",["1 - Low","2 - Medium","3 - High","4 - Very High"])
job_satisfaction = int(job_satisfaction[0])

relationship_satisfaction = st.sidebar.selectbox("**Relationship Satisfaction**",["1 - Low","2 - Medium","3 - High","4 - Very High"])
relationship_satisfaction = int(relationship_satisfaction[0])

work_life_balance = st.sidebar.selectbox("**Work Life Balance**",["1 - Low","2 - Medium","3 - High","4 - Very High"])
work_life_balance = int(work_life_balance[0])

job_involvement = st.sidebar.selectbox("**Job Involvement**",["1 - Low","2 - Medium","3 - High","4 - Very High"])
job_involvement = int(job_involvement[0])

st.sidebar.markdown("---")

st.sidebar.subheader("📈 **Performance**")

performance_rating = st.sidebar.selectbox("**Performance Rating**",["3 - Excellent","4 - OutStanding"])
performance_rating = int(performance_rating[0])

training_times_last_year = st.sidebar.selectbox("**Training Times Last Year**",[0,1,2,3,4,5,6],help="Number of training programs attended last year")

st.sidebar.markdown("---")

st.sidebar.subheader("🏢 **Experience**")

total_working_years = st.sidebar.number_input(
    "**Total Working Years**",
    min_value=0,
    max_value=50,
    value=10
)

years_at_company = st.sidebar.number_input(
    "**Years At Company**",
    min_value=0,
    max_value=40,
    value=5
)

years_in_current_role = st.sidebar.number_input(
    "**Years In Current Role**",
    min_value=0,
    max_value=30,
    value=3
)

years_since_last_promotion = st.sidebar.number_input(
    "**Years Since Last Promotion**",
    min_value=0,
    max_value=20,
    value=1
)

years_with_current_manager = st.sidebar.number_input(
    "**Years With Current Manager**",
    min_value=0,
    max_value=20,
    value=3
)

num_companies_worked = st.sidebar.number_input(
    "**Number of Companies Worked**",
    min_value=0,
    max_value=10,
    value=1
)


st.sidebar.markdown("---")

st.sidebar.subheader("✈️ **Travel**")

distance_from_home = st.sidebar.number_input("**Distance From Home**",min_value=1,max_value=29,value=5)
business_travel = st.sidebar.selectbox("**Business Travel**",['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])

st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ **Company Standards**")

standard_hours = st.sidebar.number_input(
    "**Standard Hours**",
    value=80,
    disabled=True
)

user_input = pd.DataFrame({
    "Age" : [age],
    "Gender" : [gender],
    "MaritalStatus" : [marital_status],
    "Department" : [department],
    "JobRole" : [job_role],
    "JobLevel" : [job_level],
    "OverTime" : [overtime],
    "Education" : [education],
    "EducationField" : [education_field],
    "MonthlyIncome" : [monthly_income],
    "StockOptionLevel" : [stock_opt_level],
    "PercentSalaryHike" : [percent_salary_hike],
    "HourlyRate" : [hourly_rate],
    "DailyRate" : [daily_rate],
    "MonthlyRate" : [monthly_rate],
    "EnvironmentSatisfaction" : [environment_satisfaction],
    "JobSatisfaction" : [job_satisfaction],
    "RelationshipSatisfaction" : [relationship_satisfaction],
    "WorkLifeBalance" : [work_life_balance],
    "JobInvolvement" : [job_involvement],
    "PerformanceRating" : [performance_rating],
    "TrainingTimesLastYear" : [training_times_last_year],
    "TotalWorkingYears" : [total_working_years],
    "YearsAtCompany" : [years_at_company],
    "YearsInCurrentRole" : [years_in_current_role],
    "YearsSinceLastPromotion" : [years_since_last_promotion],
    "YearsWithCurrManager" : [years_with_current_manager],
    "NumCompaniesWorked" : [num_companies_worked],
    "DistanceFromHome" : [distance_from_home],
    "BusinessTravel" : [business_travel],
    "StandardHours":[standard_hours]
})

processed_input = preprocessor.transform(user_input)
feature_names = preprocessor.get_feature_names_out()
processed_df = pd.DataFrame(processed_input,columns=feature_names)
processed_df.columns = (
    processed_df.columns
    .str.replace("onehot__", "", regex=False)
    .str.replace("scaler__", "", regex=False)
    .str.replace("remainder__", " ", regex=False)
    .str.replace(r"(?<!^)(?=[A-Z])","",regex=True)
)


shap_values = explainer(processed_df)

probability = model.predict_proba(processed_input)

leave_probability = probability[0,1]


if st.button("🧑🏻 Predict Attrition", use_container_width=True):

    st.session_state.prediction_history.append({

    "Age": age,
    "Department": department,
    "Job Role": job_role,
    "Monthly Income": monthly_income,
    "Probability": round(leave_probability * 100, 2),
    "Prediction": "High Attrition Risk" if leave_probability >= THRESHOLD else "Low Attrition Risk"

})    

    st.subheader("👤 Employee Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Age:**", age)
        st.write("**Gender:**", gender)
        st.write("**Department:**", department)
        st.write("**Job Role:**", job_role)
        st.write("**Education:**", education)

    with col2:
        st.write("**Monthly Income:**", monthly_income)
        st.write("**Business Travel:**", business_travel)
        st.write("**OverTime:**", overtime)
        st.write("**Years at Company:**", years_at_company)
        st.write("**Job Level:**", job_level)

    st.divider()



    if leave_probability >= THRESHOLD:

        st.error("🚨 High Attrition Risk")

        st.markdown("""
### Employee is likely to leave
""")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Attrition Probability",
                f"{leave_probability:.2%}"
            )

        with col2:
            st.metric(
                "Decision Threshold",
                f"{THRESHOLD:.0%}"
            )

        with col3:
            st.metric(
                "Confidence",
                f"{max(leave_probability, 1 - leave_probability):.2%}"
            )

        st.markdown("""
**Recommendation**

- Schedule HR discussion
- Review workload
- Review compensation & satisfaction
""")

    else:

        st.success("✅ Low Attrition Risk")

        st.markdown("""
### Employee is likely to stay
""")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Attrition Probability",
                f"{leave_probability:.2%}"
            )

        with col2:
            st.metric(
                "Decision Threshold",
                f"{THRESHOLD:.0%}"
            )

        with col3:
            st.metric(
                "Confidence",
                f"{max(leave_probability, 1 - leave_probability):.2%}"
            )

        st.markdown("""
**Recommendation**

- Continue normal monitoring

- Employee appears stable
""")
        
    st.subheader("📈 Attrition Risk Meter")  
    st.progress(float(leave_probability))  
    st.write(f"Risk Score: **{leave_probability:.2%}**")

        
    st.divider()

    st.subheader("⭐ Why did the model make this prediction?")

    fig = plt.figure(figsize=(10, 5))

    shap.plots.waterfall(
        shap_values[0],
        max_display=10,
        show=False
    )

    st.pyplot(fig)    
        

st.divider()

st.subheader("📋 Prediction History")

if len(st.session_state.prediction_history) > 0:
    history_df = pd.DataFrame(st.session_state.prediction_history)
    
    st.dataframe(history_df, use_container_width=True)
    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction History",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.info("No predictions made yet.")

st.divider()

st.header("📂 Bulk Employee Prediction")

uploaded_file = st.file_uploader(
    "Upload Employee CSV",
    type=["csv"]
)

if uploaded_file is not None:

    uploaded_df = pd.read_csv(uploaded_file)

    st.success("✅ File uploaded successfully!")

    processed_bulk = preprocessor.transform(uploaded_df)

    bulk_probability = model.predict_proba(processed_bulk)

    leave_probability = bulk_probability[:, 1]

    bulk_prediction = [
        "High Attrition Risk" if prob >= THRESHOLD else "Low Attrition Risk"
        for prob in leave_probability
    ]

    uploaded_df["Attrition Probability (%)"] = (leave_probability * 100).round(2)
    uploaded_df["Prediction"] = bulk_prediction

    total_employees = len(uploaded_df)

    high_risk = (uploaded_df["Prediction"] == "High Attrition Risk").sum()
    low_risk = (uploaded_df["Prediction"] == "Low Attrition Risk").sum()
    average_probability = uploaded_df["Attrition Probability (%)"].mean()

    pie_df = pd.DataFrame({
    "Risk": ["High Risk", "Low Risk"],
    "Employees": [high_risk, low_risk]
   })

    fig = px.pie(
       pie_df,
       values="Employees",
       names="Risk",
       title="Employee Risk Distribution",
       hole=0.45
    )

    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(
      uploaded_df,
      x="Attrition Probability (%)",
      nbins=10,
      title="Distribution of Attrition Probability"
  )

    st.plotly_chart(fig, use_container_width=True)

    if "Department" in uploaded_df.columns:

      dept = (
          uploaded_df[uploaded_df["Prediction"] == "High Attrition Risk"]
          .groupby("Department")
          .size()
          .reset_index(name="Employees")
      )

      fig = px.bar(
          dept,
          x="Department",
          y="Employees",
          title="High Risk Employees by Department"
      )

      st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Bulk Prediction Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "👥 Total Employees",
            total_employees
        )

    with col2:
        st.metric(
            "🔴 High Risk",
            high_risk
        )

    with col3:
        st.metric(
            "🟢 Low Risk",
            low_risk
        )

    with col4:
        st.metric(
            "📈 Avg Risk",
            f"{average_probability:.2f}%"
        )

    st.divider()

    st.subheader("📊 Prediction Results")
    st.dataframe(uploaded_df, use_container_width=True)

    st.divider()

    st.subheader("🚨 High Risk Employees")

    high_risk_df = uploaded_df[uploaded_df["Prediction"] == "High Attrition Risk"]
    high_risk_df = high_risk_df.sort_values(by="Attrition Probability (%)",ascending=False)

    if len(high_risk_df) > 0:
        st.dataframe(high_risk_df,use_container_width=True)
    else:
        st.success("🎉 No hugh-risk employee found")

    csv = uploaded_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction Results",
        data=csv,
        file_name="employee_attrition_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray;">
        Built by <b>Sarthak Maurya</b> ❤️ | IBM Employee Attrition Prediction
    </div>
    """,
    unsafe_allow_html=True
)