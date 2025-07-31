import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# Title and description
st.title("EduRank: MOORA-Based Stock Selection for Educational Innovation")
st.markdown("""
This app evaluates and ranks stocks based on multiple criteria using the **MOORA (Multi-Objective Optimization on the Basis of Ratio Analysis)** method. 
Upload your stock dataset, specify weights for each criterion, and the system will compute rankings based on the MOORA method.
""") 

# File uploader for decision matrix (stock data)
uploaded_file = st.file_uploader("Upload Excel or CSV file with stock data", type=["csv", "xlsx"])

# Example fallback dataset
def load_example():
    data = {
        'Stock': ['A', 'B', 'C'],
        'Price': [100, 120, 95],
        'P/E Ratio': [15, 18, 12],
        'Dividend Yield': [2.5, 3.0, 2.8],
        'Growth Rate': [8, 7, 9]
    }
    return pd.DataFrame(data)

# Load the data
df = None
if uploaded_file:
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
else:
    st.info("No file uploaded. Using example dataset.")
    df = load_example()

# Display the uploaded data or example
st.subheader("Stock Data")
st.dataframe(df)

# Extract stock names and criteria
stocks = df.iloc[:, 0]
criteria = df.columns[1:]
data = df.iloc[:, 1:].astype(float)

# Define Benefit and Cost Criteria (manual inputs for this example)
benefit_criteria = ['EPS', 'DPS', 'NTA', 'DY', 'ROE']  # Update based on your dataset
cost_criteria = ['PE', 'PTBV']  # Update based on your dataset

# Normalize the data using vector normalization
st.subheader("Step 1: Normalize the Data")
normalized = data.copy()
for i, col in enumerate(criteria):
    norm = data[col] / np.sqrt((data[col]**2).sum())
    normalized[col] = norm
st.dataframe(normalized)

# Step 2: Sorting Normalized Values based on Benefit and Cost Criteria
st.subheader("Step 2: Sort Normalized Values by Criteria (Benefit or Cost)")
benefit_data = normalized[benefit_criteria]
cost_data = normalized[cost_criteria]

# Display the normalized values separated by Benefit and Cost Criteria
st.write("Benefit Criteria")
st.dataframe(benefit_data)

st.write("Cost Criteria")
st.dataframe(cost_data)

# Step 3: Calculate Benefit Minus Cost (For PIS and NIS Calculation)
st.subheader("Step 3: Calculate Benefit Minus Cost")

# Calculate total benefit minus total cost for each alternative
benefit_minus_cost = benefit_data.sum(axis=1) - cost_data.sum(axis=1)

# Add the result to the data
normalized['Benefit - Cost'] = benefit_minus_cost
st.dataframe(normalized)

# Step 4: Rank the Alternatives Based on the Calculated Scores
st.subheader("Step 4: Final Rankings")
normalized['Rank'] = normalized['Benefit - Cost'].rank(ascending=False)
st.dataframe(normalized)

# Download Results as CSV
st.subheader("Download Result")
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(normalized)
st.download_button("Download Results as CSV", csv, "edurank_results.csv", "text/csv")
