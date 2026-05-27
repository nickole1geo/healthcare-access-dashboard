import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.autolayout": True
})

# PAGE TITLE
st.title("Healthcare Access Dashboard Using NY SPARCS Data")

st.write("""
This dashboard explores potentially avoidable ED-related utilization
using de-identified NY SPARCS hospital discharge data.
""")

# LOAD DATA
url = "https://health.data.ny.gov/resource/5dtw-tffi.csv?$limit=50000"

df = pd.read_csv(url)

# CLEAN COLUMN NAMES
df.columns = (
    df.columns
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("/", "_")
    .str.replace("-", "_")
)

# FILTER TO ED CASES
df = df[df["emergency_department_indicator"].str.upper() == "Y"].copy()

# CREATE OUTCOME VARIABLE
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

# TITLE
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Insurance Analysis", "Geography", "Machine Learning"])
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
    st.pyplot(fig1)

    st.subheader("Potentially Avoidable ED Rate by Insurance Type")

    payer_summary = (
        filtered_df.groupby("payment_typology_1")["avoidable_ed"]
        .mean()
        .sort_values(ascending=False)
    )

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    payer_summary.plot(kind="bar", ax=ax2)
    ax2.set_ylabel("Avoidable ED Rate")
    ax2.set_xlabel("Primary Payer")
    st.pyplot(fig2)

with tab3:
    st.subheader("Top Counties by ED-Related Discharges")

    county_counts = filtered_df["hospital_county"].value_counts().head(15)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    county_counts.plot(kind="bar", ax=ax3)
    ax3.set_ylabel("Number of ED-Related Discharges")
    ax3.set_xlabel("Hospital County")
    st.pyplot(fig3)

    st.subheader("Potentially Avoidable ED Rate by County")

    county_avoidable = (
        filtered_df.groupby("hospital_county")["avoidable_ed"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    county_avoidable.plot(kind="bar", ax=ax4)
    ax4.set_ylabel("Avoidable ED Rate")
    ax4.set_xlabel("Hospital County")
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
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)

    st.metric("ROC AUC Score", f"{auc:.3f}")

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax5)
    ax5.set_title("ROC Curve: Avoidable ED Prediction Model")
    st.pyplot(fig5)

    st.subheader("Top Predictors of Potentially Avoidable ED Classification")

    import numpy as np

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

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax6.barh(top_features["feature"], top_features["coefficient"])
    ax6.set_xlabel("Model Coefficient")
    ax6.set_title("Top Model Predictors")
    st.pyplot(fig6)

    st.dataframe(
        top_features[["feature", "coefficient", "odds_ratio"]]
    )
