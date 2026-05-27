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

# TITLE
st.subheader("Discharges by Primary Payer")

# GRAPH
fig1, ax1 = plt.subplots()

df["payment_typology_1"].value_counts().head(10).plot(
    kind="bar",
    ax=ax1
)

ax1.set_ylabel("Count")
ax1.set_xlabel("Primary Payer")

st.pyplot(fig1)

# SECOND GRAPH
st.subheader("Potentially Avoidable ED Rate by Insurance Type")

payer_summary = (
    df.groupby("payment_typology_1")["avoidable_ed"]
    .mean()
    .sort_values(ascending=False)
)

fig2, ax2 = plt.subplots()

payer_summary.plot(
    kind="bar",
    ax=ax2
)

ax2.set_ylabel("Avoidable ED Rate")
ax2.set_xlabel("Primary Payer")

st.pyplot(fig2)
