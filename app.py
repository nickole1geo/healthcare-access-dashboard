import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
tab1, tab2, tab3 = st.tabs(["Overview", "Insurance Analysis", "Geography"])
with tab1:
    st.subheader("Project Overview")
    st.write("""
    This dashboard uses de-identified NY SPARCS hospital discharge data to explore
    potentially avoidable ED-related utilization. Avoidable ED use is operationalized
    as cases classified as minor severity of illness.
    """)

with tab2:
    st.subheader("Discharges by Primary Payer")

    fig1, ax1 = plt.subplots()
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

    fig2, ax2 = plt.subplots()
    payer_summary.plot(kind="bar", ax=ax2)
    ax2.set_ylabel("Avoidable ED Rate")
    ax2.set_xlabel("Primary Payer")
    st.pyplot(fig2)

with tab3:
    st.subheader("Top Counties by ED-Related Discharges")

    county_counts = filtered_df["hospital_county"].value_counts().head(15)

    fig3, ax3 = plt.subplots()
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

    fig4, ax4 = plt.subplots()
    county_avoidable.plot(kind="bar", ax=ax4)
    ax4.set_ylabel("Avoidable ED Rate")
    ax4.set_xlabel("Hospital County")
    st.pyplot(fig4)
