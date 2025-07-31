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

# Input weights for each criterion
st.subheader("Input Weights (must sum to 1)")
weights = []
for i, col in enumerate(criteria):
    weight = st.number_input(f"Weight for {col}", min_value=0.0, max_value=1.0, value=1/len(criteria), step=0.01)
    weights.append(weight)

# Ensure weights sum to 1
if sum(weights) != 1:
    st.warning("Weights must sum to 1! Please adjust the weights.")

# Normalize the data using vector normalization
st.subheader("Step 1: Normalize the Data")
normalized = data.copy()
for i, col in enumerate(criteria):
    norm = data[col] / np.sqrt((data[col]**2).sum())
    normalized[col] = norm
st.dataframe(normalized)

# Sort the normalized values based on their criteria (cost or benefit)
st.subheader("Step 2: Sort Normalized Values by Criteria (Cost or Benefit)")
sorted_data = normalized.copy()
benefit_criteria = []  # Assuming you have benefit criteria manually defined here
cost_criteria = []  # Assuming you have cost criteria manually defined here

# Displaying the sorted normalized matrix (based on whether it's benefit or cost)
# Sort by Benefit Criteria first
for col in criteria:
    if col in benefit_criteria:
        sorted_data = sorted_data.sort_values(by=col, ascending=False)
    elif col in cost_criteria:
        sorted_data = sorted_data.sort_values(by=col, ascending=True)

st.dataframe(sorted_data)

# Step 3: Calculate Benefit Minus Cost (For PIS and NIS Calculation)
st.subheader("Step 3: Calculate Benefit Minus Cost")

# Assuming benefit criteria are to be subtracted by cost criteria
benefit_columns = [col for col in criteria if col in benefit_criteria]
cost_columns = [col for col in criteria if col in cost_criteria]

# Calculate total benefit minus total cost for each alternative
benefit_minus_cost = sorted_data[benefit_columns].sum(axis=1) - sorted_data[cost_columns].sum(axis=1)

# Add the result to the data
sorted_data['Benefit - Cost'] = benefit_minus_cost
st.dataframe(sorted_data)

# Step 4: Rank the Alternatives Based on the Calculated Scores
st.subheader("Step 4: Final Rankings")
sorted_data['Rank'] = sorted_data['Benefit - Cost'].rank(ascending=False)
st.dataframe(sorted_data)

# Download Results as CSV
st.subheader("Download Result")
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(sorted_data)
st.download_button("Download Results as CSV", csv, "edurank_results.csv", "text/csv")
