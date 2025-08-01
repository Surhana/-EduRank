import streamlit as st
import pandas as pd
import numpy as np

st.title("EduRank: MOORA-Based Stock Selection for Educational Innovation")

st.markdown("""
Upload your stock dataset, then select which columns are **Benefit** and **Cost** criteria.
The app will normalize your data, apply **weights**, calculate **Benefit - Cost** scores, and rank the alternatives.
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

if benefit_criteria or cost_criteria:

    # Step 1: Normalize the data
    st.subheader("Step 1: Normalize the Data")
    normalized = numeric_df.copy()
    for col in criteria_cols:
        normalized[col] = numeric_df[col] / np.sqrt((numeric_df[col]**2).sum())
    st.dataframe(normalized)

    # Step 2: Weighted Normalized Matrix
    st.subheader("Step 2: Weighted Normalized Matrix (Benefit & Cost)")
    weights = []
    st.write("Enter weights for each selected criterion (must sum to 1):")
    for col in benefit_criteria + cost_criteria:
        w = st.number_input(f"Weight for {col}", min_value=0.0, max_value=1.0, value=1.0/len(benefit_criteria+cost_criteria), step=0.01)
        weights.append(w)

    # Convert to numpy array for calculations
    weights = np.array(weights)

    # Create weighted normalized dataframe
    selected_criteria = benefit_criteria + cost_criteria
    weighted_normalized = normalized[selected_criteria].copy()
    for i, col in enumerate(selected_criteria):
        weighted_normalized[col] = weighted_normalized[col] * weights[i]

    # Display weighted normalized matrix
    st.write("Weighted Normalized Matrix:")
    st.dataframe(weighted_normalized)

    # Step 3: Calculate Benefit - Cost using weighted normalized values
    st.subheader("Step 3: Calculate Benefit Minus Cost (MOORA Score)")
    benefit_data = weighted_normalized[benefit_criteria] if benefit_criteria else pd.DataFrame(np.zeros((len(stocks),0)))
    cost_data = weighted_normalized[cost_criteria] if cost_criteria else pd.DataFrame(np.zeros((len(stocks),0)))

    score = benefit_data.sum(axis=1) - cost_data.sum(axis=1)

    result = pd.DataFrame({
        'Stock': stocks,
        'Benefit - Cost': score
    })
    result['Benefit - Cost'] = result['Benefit - Cost'].round(4)  # 4 decimal places
    st.dataframe(result)

    # Step 4: Rank (as 1,2,3...)
    st.subheader("Step 4: Final Rankings")
    result = result.sort_values('Benefit - Cost', ascending=False).reset_index(drop=True)
    result['Rank'] = range(1, len(result) + 1)  # 1,2,3...

    # Highlight the top-ranked alternative in green
    def highlight_top(row):
        return ['background-color: lightgreen'] * len(row) if row.name == 0 else [''] * len(row)

    st.dataframe(result.style.apply(highlight_top, axis=1))

    # Announce the best alternative
    best_stock = result.loc[0, 'Stock']
    st.success(f"🏆 **The Best Alternative is:** {best_stock} 🎉💹")

    # Download CSV
    csv = result.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="edurank_results.csv",
        mime="text/csv"
    )
else:
    st.warning("Please select at least one benefit or cost criterion to continue.")
