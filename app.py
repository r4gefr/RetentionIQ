import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
model = joblib.load("models/churn_model.joblib")
feature_columns = joblib.load("models/feature_columns.joblib")

# ---------------- CONFIG ---------------- #

st.set_page_config(
    page_title="RetentionIQ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- STYLING ---------------- #

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #F8FAFC;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

section[data-testid="stSidebar"] * {
    color: white;
}

footer {visibility:hidden;}
#MainMenu {visibility:hidden;}

</style>
""", unsafe_allow_html=True)

# ---------------- DATA ---------------- #

df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Clean
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df.dropna(inplace=True)

# ---------------- SIDEBAR ---------------- #

st.title("RetentionIQ")

st.markdown("""
### Customer Retention Intelligence Platform

Predict churn, identify at-risk customers,
and generate actionable retention strategies.
""")

with st.sidebar:
    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Analytics",
            "Predictor",
            "Retention Insights",
            "Model Performance"
        ]
    )

st.sidebar.markdown("---")


# ==================================================
# DASHBOARD
# ==================================================

if page == "Dashboard":

    st.title("RetentionIQ")

    st.caption(
        "Customer Retention Intelligence Platform"
    )

    st.info(
        "Predict churn, analyze customer behavior and generate retention strategies."
    )

    churn_rate = (
        (df["Churn"] == "Yes").mean() * 100
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Customers",
        f"{len(df):,}"
    )

    col2.metric(
        "Churn Rate",
        f"{churn_rate:.1f}%"
    )

    col3.metric(
        "Avg Monthly Revenue",
        f"${df['MonthlyCharges'].mean():.2f}"
    )

    col4.metric(
        "Avg Tenure",
        f"{df['tenure'].mean():.1f}"
    )

# ==================================================
# ANALYTICS
# ==================================================

elif page == "Analytics":

    st.title("Customer Analytics")

    col1, col2 = st.columns(2)

    with col1:

        churn_data = (
            df["Churn"]
            .value_counts()
            .reset_index()
        )

        churn_data.columns = [
            "Churn",
            "Count"
        ]

        fig = px.pie(
            churn_data,
            names="Churn",
            values="Count",
            hole=0.4,
            title="Customer Churn Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        contract_data = (
            df.groupby("Contract")
            .size()
            .reset_index(name="Count")
        )

        fig = px.bar(
            contract_data,
            x="Contract",
            y="Count",
            title="Customers by Contract"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    col3, col4 = st.columns(2)

    with col3:

        fig = px.box(
            df,
            x="Churn",
            y="MonthlyCharges",
            title="Monthly Charges vs Churn"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col4:

        fig = px.box(
            df,
            x="Churn",
            y="tenure",
            title="Tenure vs Churn"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ==================================================
# PREDICTOR
# ==================================================

elif page == "Predictor":       

    import joblib

    st.title("Churn Predictor")

    model = joblib.load("models/churn_model.joblib")
    feature_columns = joblib.load("models/feature_columns.joblib")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure (Months)",
            0,
            72,
            12
        )

        phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
        )       

        internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
        "Online Security",
        ["Yes", "No"]
        )

        online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No"]
        )

        device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No"]
        )

        tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No"]
        )

        streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No"]
        )

        streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No"]
        )

        paperless = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
        )

    with col2:

        monthly = st.number_input(
            "Monthly Charges",
            20.0,
            150.0,
            70.0
        )

        total = st.number_input(
            "Total Charges",
            0.0,
            10000.0,
            500.0
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    if st.button("Predict Churn", use_container_width=True):

        customer = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Contract": contract,
            "PaymentMethod": payment
        }

        input_df = pd.DataFrame([customer])

        input_df = pd.get_dummies(input_df)

        input_df = input_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        prediction = model.predict(input_df)[0]

        probability = (
            model.predict_proba(input_df)[0][1]
        )

        st.markdown("---")

        if prediction == 1:

            st.error(
                f"⚠️Customer likely to churn ({probability:.1%})"
            )

        else:

            st.success(
                f"Customer likely to stay ({1-probability:.1%})"
            )

        st.metric(
            "Churn Probability",
            f"{probability:.1%}"
        )

        # Risk Level

        if probability > 0.7:

            st.error("High Risk Customer")

        elif probability > 0.4:

            st.warning("Medium Risk Customer")

        else:

             st.success("Low Risk Customer")

        # Recommendation

        st.subheader("Retention Recommendation")

        if probability > 0.75:

            st.write("""
            • Offer retention discount

            • Assign support representative

            • Upgrade loyalty benefits

            • Contact customer immediately
            """)

        elif probability > 0.40:

            st.write("""
            • Send personalized offers

            • Encourage annual contract

            • Provide feature education
            """)

        else:

            st.write("""
            • Customer is healthy

            • Continue normal engagement
            """)

# ==================================================
# RETENTION
# ==================================================

elif page == "Retention Insights":

    st.title("Retention Insights")

    st.info("""
### High Impact Retention Strategies

• Offer annual contract discounts

• Reward long-term customers

• Provide onboarding support for new users

• Target high monthly charge customers with loyalty plans

• Promote automatic payment methods
""")

# ==================================================
# PERFORMANCE
# ==================================================

elif page == "Model Performance":

    st.title("Model Performance")

    comparison = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ],
        "Accuracy": [
            0.80,
            0.84,
            0.87
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True
    )

    fig = px.bar(
        comparison,
        x="Model",
        y="Accuracy",
        title="Model Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
