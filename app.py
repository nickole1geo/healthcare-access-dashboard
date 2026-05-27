import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

st.set_page_config(layout="wide")

# TITLE
st.title("Healthcare Access Dashboard Using NY SPARCS Data")

st.write("""
This dashboard explores potentially avoidable ED-related utilization
using de-identified NY SPARCS hospital discharge data.
""")

# LOAD DATA
url = "https://health.data.ny.gov/resource/5dtw-tffi.csv?$limit=50000"

df = pd.read_csv(url, low_memory=False)

# CLEAN DATA
df.columns = df.columns.str.lower()

# CREATE AVOIDABLE ED VARIABLE
df["avoidable_ed"] = np.where(
    df["apr_severity_of_illness_description"] == "Minor",
    1,
    0
)

# FILTERS
insurance_options = sorted(df["payment_typology_1"].dropna().unique())

selected_insurance = st.sidebar.selectbox(
    "Select Insurance Type",
    ["All"] + insurance_options
)

if selected_insurance != "All":
    filtered_df = df[df["payment_typology_1"] == selected_insurance]
else:
    filtered_df = df.copy()

# KEY METRICS
st.header("Key Metrics")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Total ED-Related Cases",
        len(filtered_df)
    )

with col2:
    avoidable_rate = round(
        filtered_df["avoidable_ed"].mean() * 100,
        1
    )

    st.metric(
        "Potentially Avoidable ED Rate",
        f"{avoidable_rate}%"
    )

# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Insurance Analysis",
    "Machine Learning",
    "Results & Discussion",
    "Mental Health Analysis"
])

# OVERVIEW TAB
with tab1:

    st.header("Project Overview")

    st.write("""
    This project evaluates potentially avoidable emergency department utilization
    using NY SPARCS inpatient discharge data.
    """)

    st.subheader("Methods")

    st.write("""
    Avoidable ED utilization was operationalized using APR Severity of Illness.
    Visits categorized as 'Minor' severity were classified as potentially avoidable.
    """)

    st.subheader("Limitations")

    st.write("""
    Severity of illness is a proxy measure and does not represent a definitive
    clinical determination of avoidability.
    """)

# INSURANCE TAB
with tab2:

    st.header("Insurance Analysis")

    payer_summary = (
        filtered_df.groupby("payment_typology_1")["avoidable_ed"]
        .mean()
        .sort_values(ascending=False)
    )

    fig1, ax1 = plt.subplots(figsize=(10,5))

    payer_summary.plot(kind="bar", ax=ax1)

    ax1.set_ylabel("Avoidable ED Rate")
    ax1.set_xlabel("Insurance Type")
    ax1.tick_params(axis="x", rotation=45)

    st.pyplot(fig1)

    st.subheader("Interpretation")

    st.write("""
    Differences across insurance categories may reflect variation in
    outpatient access, care coordination, referral systems,
    and healthcare utilization patterns.
    """)

# MACHINE LEARNING TAB
with tab3:

    st.header("Machine Learning Prediction Model")

    ml_df = filtered_df.copy()

    features = [
        "age_group",
        "gender",
        "payment_typology_1"
    ]

    ml_df = ml_df.dropna(subset=features)

    X = pd.get_dummies(
        ml_df[features],
        drop_first=True
    )

    y = ml_df["avoidable_ed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    st.metric(
        "Model Accuracy",
        round(accuracy, 3)
    )

    coef_df = pd.DataFrame({
        "feature": X.columns,
        "coefficient": model.coef_[0]
    })

    coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])

    top_features = coef_df.sort_values(
        by="odds_ratio",
        ascending=False
    ).head(10)

    st.subheader("Top Predictive Features")

    st.dataframe(top_features)

# RESULTS TAB
with tab4:

    st.header("Results & Discussion")

    st.subheader("Key Findings")

    st.markdown("""
    - Avoidable ED utilization varied across insurance groups
    - Mental health-related visits showed important disparities
    - Logistic regression identified demographic and insurance-related predictors
    - Results suggest structural barriers in outpatient access
    """)

    st.subheader("Policy Implications")

    st.markdown("""
    - Expand outpatient preventive care access
    - Improve behavioral healthcare infrastructure
    - Increase insurance network adequacy
    - Strengthen care coordination systems
    - Improve crisis stabilization resources
    """)

    st.subheader("Future Directions")

    st.markdown("""
    - Add geographic hotspot mapping
    - Build advanced ML models
    - Develop predictive risk stratification
    - Compare across multiple years of SPARCS data
    - Integrate AI-driven forecasting approaches
    """)

# MENTAL HEALTH TAB
with tab5:

    st.header("Mental Health Analysis")

    df["mental_health_dx"] = (
        df["apr_mdc_description"]
        .astype(str)
        .str.lower()
        .str.contains("mental", na=False)
    )

    mh_summary = (
        df.groupby("mental_health_dx")["avoidable_ed"]
        .mean()
    )

    mh_summary.index = [
        "No Mental Health Diagnosis",
        "Mental Health Diagnosis"
    ]

    fig2, ax2 = plt.subplots(figsize=(8,5))

    mh_summary.plot(kind="bar", ax=ax2)

    ax2.set_ylabel("Avoidable ED Rate")
    ax2.tick_params(axis="x", rotation=0)

    st.pyplot(fig2)

    mh_df = df[df["mental_health_dx"] == True]

    st.subheader("Avoidable ED Rate by Insurance Type Among Mental Health Visits")

    if len(mh_df) > 0:

        mh_payer_summary = (
            mh_df.groupby("payment_typology_1")["avoidable_ed"]
            .mean()
            .sort_values(ascending=False)
        )

        fig3, ax3 = plt.subplots(figsize=(10,5))

        mh_payer_summary.plot(kind="bar", ax=ax3)

        ax3.set_ylabel("Avoidable ED Rate")
        ax3.set_xlabel("Insurance Type")
        ax3.tick_params(axis="x", rotation=45)

        st.pyplot(fig3)

    st.subheader("Interpretation")

    st.write("""
    Mental health-related ED utilization may reflect barriers in
    outpatient psychiatric access, crisis intervention availability,
    and continuity of behavioral healthcare.
    """)
