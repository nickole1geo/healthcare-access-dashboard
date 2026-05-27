import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.autolayout": True})

st.title("Healthcare Access Dashboard Using NY SPARCS Data")

st.write("""
This dashboard explores potentially avoidable ED-related utilization
using de-identified NY SPARCS hospital discharge data.
""")

url = "https://health.data.ny.gov/resource/5dtw-tffi.csv?$limit=50000"
df = pd.read_csv(url)

df.columns = (
    df.columns
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("/", "_")
    .str.replace("-", "_")
)

df = df[df["emergency_department_indicator"].str.upper() == "Y"].copy()

df["avoidable_ed"] = (
    df["apr_severity_of_illness"]
    .str.lower()
    .eq("minor")
    .astype(int)
)

st.subheader("Key Metrics")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total ED-Related Cases", len(df))

with col2:
    st.metric("Potentially Avoidable ED Rate", f"{df['avoidable_ed'].mean():.1%}")

st.sidebar.header("Filters")

payer_options = df["payment_typology_1"].dropna().unique()

selected_payer = st.sidebar.selectbox(
    "Select Insurance Type",
    ["All"] + list(payer_options)
)

if selected_payer == "All":
    filtered_df = df.copy()
else:
    filtered_df = df[df["payment_typology_1"] == selected_payer]

st.write(f"Current filter: {selected_payer}")
st.write(f"Records shown: {len(filtered_df)}")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Insurance Analysis",
    "Geography",
    "Machine Learning",
    "Results & Discussion",
    "Mental Health Analysis"
])

with tab1:
    st.subheader("Project Overview")
    st.write("""
    This dashboard uses de-identified NY SPARCS hospital discharge data to explore
    potentially avoidable ED-related utilization. Avoidable ED use is operationalized
    as cases classified as minor severity of illness.
    """)

    st.subheader("Methods")
    st.write("""
    The analysis uses the public de-identified NY SPARCS hospital discharge dataset.
    Records were filtered to ED-related discharges using the emergency department indicator.
    Potentially avoidable ED-related utilization was operationalized as cases classified
    as minor severity of illness using APR severity.
    """)

    st.subheader("Limitations")
    st.write("""
    This dashboard is exploratory and should not be interpreted as causal evidence.
    Because avoidable ED use is operationalized using severity of illness, the model
    predicts a proxy measure rather than a definitive clinical determination of avoidability.
    Results may also be influenced by coding practices, diagnosis mix, and differences in
    hospital reporting.
    """)

with tab2:
    st.subheader("Discharges by Primary Payer")

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    filtered_df["payment_typology_1"].value_counts().head(10).plot(
        kind="bar",
        ax=ax1
    )
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Primary Payer")
    ax1.tick_params(axis="x", rotation=45)
    st.pyplot(fig1)

    st.subheader("Potentially Avoidable ED Rate by Insurance Type")

    payer_summary = (
        filtered_df.groupby("payment_typology_1")["avoidable_ed"]
        .mean()
        .sort_values(ascending=False)
    )

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    payer_summary.plot(kind="bar", ax=ax2)
    ax2.set_ylabel("Avoidable ED Rate")
    ax2.set_xlabel("Primary Payer")
    ax2.tick_params(axis="x", rotation=45)
    st.pyplot(fig2)

with tab3:
    st.subheader("Top Counties by ED-Related Discharges")

    county_counts = filtered_df["hospital_county"].value_counts().head(15)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    county_counts.plot(kind="bar", ax=ax3)
    ax3.set_ylabel("Number of ED-Related Discharges")
    ax3.set_xlabel("Hospital County")
    ax3.tick_params(axis="x", rotation=45)
    st.pyplot(fig3)

    st.subheader("Potentially Avoidable ED Rate by County")

    county_avoidable = (
        filtered_df.groupby("hospital_county")["avoidable_ed"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )

    fig4, ax4 = plt.subplots(figsize=(10, 5))
    county_avoidable.plot(kind="bar", ax=ax4)
    ax4.set_ylabel("Avoidable ED Rate")
    ax4.set_xlabel("Hospital County")
    ax4.tick_params(axis="x", rotation=45)
    st.pyplot(fig4)

with tab4:
    st.subheader("Machine Learning Model")

    st.write("""
    This exploratory model predicts whether an ED-related discharge is classified
    as potentially avoidable, using patient demographics, payer, hospital county,
    and diagnosis category.
    """)

    features = [
        "age_group",
        "gender",
        "race",
        "ethnicity",
        "payment_typology_1",
        "hospital_county",
        "ccsr_diagnosis_description"
    ]

    model_df = df[features + ["avoidable_ed"]].dropna()

    X = pd.get_dummies(model_df[features], dummy_na=True)
    y = model_df["avoidable_ed"]

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, RocCurveDisplay

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    st.metric("ROC AUC Score", f"{auc:.3f}")

    fig5, ax5 = plt.subplots(figsize=(10, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax5)
    ax5.set_title("ROC Curve: Avoidable ED Prediction Model")
    st.pyplot(fig5)

    st.subheader("Top Predictors of Potentially Avoidable ED Classification")

    coef_df = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient": model.coef_[0],
        "odds_ratio": np.exp(model.coef_[0])
    })

    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()

    top_features = (
        coef_df.sort_values("abs_coefficient", ascending=False)
        .head(15)
        .sort_values("coefficient")
    )

    fig6, ax6 = plt.subplots(figsize=(10, 5))
    ax6.barh(top_features["feature"], top_features["coefficient"])
    ax6.set_xlabel("Model Coefficient")
    ax6.set_title("Top Model Predictors")
    st.pyplot(fig6)

    st.dataframe(
        top_features[["feature", "coefficient", "odds_ratio"]]
    )

with tab5:
    st.subheader("Key Results")

    st.markdown("""
    ### 1. Avoidable ED use varies by insurance type
    In the research analysis, avoidable ED visits accounted for 18.3% of ED-related encounters overall.
    Rates were highest among privately insured patients and uninsured patients, and lower among publicly insured patients.

    ### 2. The insurance story changes after adjustment
    Public insurance initially appeared strongly protective in the crude model. After adjusting for demographics,
    geography, and clinical factors, public insurance still showed lower odds of avoidable ED use, but the effect became smaller.

    ### 3. Mental health changes the pattern
    Among visits with a mental health diagnosis, the public-insurance protective effect became stronger,
    while the uninsured association reversed direction and became associated with higher odds of avoidable classification.

    ### 4. Geography matters
    The uninsured pattern differed between NYC and Non-NYC facilities, suggesting that safety-net infrastructure,
    access barriers, and local care systems may shape ED use differently across regions.
    """)

    st.subheader("Interpretation")

    st.write("""
    These findings suggest that potentially avoidable ED use should not be framed simply as patient overuse.
    Instead, ED use may reflect structural barriers in primary care access, specialty referral systems,
    mental health access, insurance design, and safety-net availability.
    """)

    st.subheader("Policy and Health System Implications")

    st.markdown("""
    - Expand same-day and after-hours primary care access.
    - Strengthen behavioral health access, especially for privately insured and uninsured patients.
    - Improve specialty referral pathways so patients do not need to use the ED as a workaround.
    - Compare NYC and Non-NYC safety-net structures to understand why uninsured patterns differ.
    - Use predictive modeling carefully as a screening and planning tool, not as a causal explanation.
    """)

    st.subheader("Next Steps for AI / Machine Learning")

    st.markdown("""
    - Compare logistic regression with random forest and gradient boosting models.
    - Add model fairness analysis by insurance type, race/ethnicity, age, and geography.
    - Add explainability tools such as SHAP values to show why the model predicts higher or lower avoidability.
    - Build a county-level risk visualization to support healthcare access planning.
    - Eventually create a decision-support prototype for identifying structural access gaps.
    """)
with tab6:
    st.subheader("Mental Health and Avoidable ED Utilization")

    st.write("""
    This section explores whether potentially avoidable ED-related utilization differs
    for visits involving mental health diagnoses. This matters because behavioral health
    access, insurance networks, and crisis-care availability may shape whether patients
    use the ED as an entry point into care.
    """)

    mental_health_terms = [
    "Mental Diseases and Disorders",
    "Mental Diseases & Disorders"
]

    df["mental_health_dx"] = df["apr_mdc_description"].isin(mental_health_terms)

    mh_summary = (
    df.groupby("mental_health_dx")["avoidable_ed"]
    .mean()
    .rename(index={
        False: "No Mental Health Diagnosis",
        True: "Mental Health Diagnosis"
    })
)

    st.subheader("Avoidable ED Rate by Mental Health Diagnosis")

    fig7, ax7 = plt.subplots(figsize=(8, 5))
    mh_summary.plot(kind="bar", ax=ax7)
    ax7.set_ylabel("Avoidable ED Rate")
    ax7.set_xlabel("")
    ax7.tick_params(axis="x", rotation=0)
    st.pyplot(fig7)

    st.subheader("Avoidable ED Rate by Insurance Type Among Mental Health Visits")

    mh_df = df[df["mental_health_dx"] == True]

    mh_payer_summary = (
        mh_df.groupby("payment_typology_1")["avoidable_ed"]
        .mean()
        .sort_values(ascending=False)
    )

    fig8, ax8 = plt.subplots(figsize=(10, 5))
    mh_payer_summary.plot(kind="bar", ax=ax8)
    ax8.set_ylabel("Avoidable ED Rate")
    ax8.set_xlabel("Primary Payer")
    ax8.tick_params(axis="x", rotation=45)
    st.pyplot(fig8)

    st.subheader("Interpretation")

    st.write("""
    Differences in avoidable ED rates among mental health-related visits may reflect
    gaps in outpatient behavioral health access, crisis stabilization options, insurance
    network adequacy, and the availability of timely psychiatric follow-up. These patterns
    should be interpreted as structural access signals rather than individual patient misuse.
    """)

    st.subheader("Policy Implications")

    st.markdown("""
    - Expand outpatient behavioral health access and same-day crisis care.
    - Improve insurance network adequacy for mental health providers.
    - Strengthen ED-to-community mental health referral pathways.
    - Evaluate whether uninsured patients face distinct barriers in NYC versus Non-NYC settings.
    - Use predictive modeling to identify where behavioral health access gaps may be most concentrated.
    """)
