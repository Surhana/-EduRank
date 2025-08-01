import streamlit as st
import pandas as pd
import numpy as np

st.title("EduRank: MOORA-Based Stock Selection for Educational Innovation")

st.markdown("""
Upload your stock dataset, then select which columns are **Benefit** and **Cost** criteria.
The app will normalize your data, calculate **Benefit - Cost** scores, and rank the alternatives.
""")

# File uploader
uploaded_file = st.file_uploader("Upload Excel or CSV file with stock data", type=["csv", "xlsx"])

# Example dataset
def load_example():
    data = {
        'Stock': ['KPJ','IHH','DPHARMA'],
        'Earnings per Share': [0.29,0.80,0.17],
        'Dividend per Share': [0.23,0.55,0.16],
        'Net Tangible Asset': [0.11,0.70,0.15],
        'Dividend Yield': [0.15,0.15,0.25],
        'Return on Equity': [0.65,0.41,0.41],
        'P/E Ratio': [0.41,0.25,0.21],
        'PTBV': [0.81,0.31,0.26]
    }
    return pd.DataFrame(data)

# Load data
if uploaded_file:
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
else:
    st.info("No file uploaded. Using example dataset.")
    df = load_example()

st.subheader("Uploaded/Example Data")
st.dataframe(df)

# Extract stock names and numeric criteria
stocks = df.iloc[:,0]
criteria_cols = df.columns[1:]
numeric_df = df[criteria_cols].astype(float)

# Step 0: Let user pick benefit and cost columns dynamically
st.subheader("Select Criteria Types")
benefit_criteria = st.multiselect("Select Benefit Criteria Columns", criteria_cols.tolist())
cost_criteria = st.multiselect("Select Cost Criteria Columns", [c for c in criteria_cols if c not in benefit_criteria])

if benefit_criteria:
    # Step 1: Normalize the data
    st.subheader("Step 1: Normalize the Data")
    normalized = numeric_df.copy()
    for col in criteria_cols:
        normalized[col] = numeric_df[col] / np.sqrt((numeric_df[col]**2).sum())
    st.dataframe(normalized)

    # Step 2: Separate tables
    st.subheader("Step 2: Separate Tables for Benefit and Cost Criteria")
    st.write("Benefit Criteria")
    st.dataframe(normalized[benefit_criteria])
    if cost_criteria:
        st.write("Cost Criteria")
        st.dataframe(normalized[cost_criteria])

    # Step 3: Calculate Benefit - Cost
    st.subheader("Step 3: Calculate Benefit Minus Cost")
    if cost_criteria:
        score = normalized[benefit_criteria].sum(axis=1) - normalized[cost_criteria].sum(axis=1)
    else:
        score = normalized[benefit_criteria].sum(axis=1)

    result = pd.DataFrame({
        'Stock': stocks,
        'Benefit - Cost': score
    })
    st.dataframe(result)

    # Step 4: Rank
    st.subheader("Step 4: Final Rankings")
    result['Rank'] = result['Benefit - Cost'].rank(ascending=False)
    result = result.sort_values('Rank')
    st.dataframe(result)

    # Download CSV
    csv = result.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results as CSV", csv, "_
