import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title("EduRank: MOORA-Based Stock Selection for Educational Innovation")
st.markdown("""
This app evaluates and ranks stocks based on multiple criteria using the **MOORA (Multi-Objective Optimization on the Basis of Ratio Analysis)** method.
Upload your stock dataset, assign weights to each criterion, and view rankings with clear tables and a bar chart.
""")

# File uploader
uploaded_file = st.file_uploader("Upload Excel or CSV file with stock data", type=["csv", "xlsx"])

# Example data
def load_example():
    data = {
        'Stock': ['A', 'B', 'C'],
        'Price': [100, 120, 95],
        'P/E Ratio': [15, 18, 12],
        'Dividend Yield': [2.5, 3.0, 2.8],
        'Growth Rate': [8, 7, 9]
    }
    return pd.DataFrame(data)

# Load data
if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)
else:
    st.info("No file uploaded. Using example dataset.")
    df = load_example()

# Ensure first column is named Alternative
df = df.rename(columns={df.columns[0]: "Alternative"})

st.subheader("Stock Data")
st.dataframe(df)

alternatives = df["Alternative"]
criteria = df.columns[1:]
data = df.iloc[:, 1:].astype(float)

# Input weights
st.subheader("Input Weights (must sum to 1)")
weights = [st.number_input(f"Weight for {col}", min_value=0.0, max_value=1.0, value=1/len(criteria), step=0.01)
           for col in criteria]

if sum(weights) != 1:
    st.warning("Weights must sum to 1! Please adjust the weights.")

# ---------------- STEP 1: Normalize ----------------
st.subheader("Step 1: Normalize the Data")
normalized_matrix = data.copy()
for i, col in enumerate(criteria):
    normalized_matrix[col] = data[col] / np.sqrt((data[col]**2).sum())

# Add Alternative column
normalized_df = pd.concat([alternatives, normalized_matrix], axis=1)
st.dataframe(normalized_df)

# ---------------- STEP 2: Weighted Normalized Matrix ----------------
st.subheader("Step 2: Weighted Normalized Matrix")
weighted_matrix = normalized_matrix.copy()
for i, col in enumerate(criteria):
    weighted_matrix[col] = normalized_matrix[col] * weights[i]

weighted_df = pd.concat([alternatives, weighted_matrix], axis=1)
st.dataframe(weighted_df)

# ---------------- STEP 3: MOORA Performance Index (PIS) ----------------
st.subheader("Step 3: MOORA Performance Index (PIS)")

# Compute PIS & NIS
positive_ideal_solution = weighted_matrix.max()
negative_ideal_solution = weighted_matrix.min()

# Show PIS & NIS
st.write("Positive Ideal Solution (PIS):")
st.dataframe(pd.DataFrame([positive_ideal_solution], columns=criteria))
st.write("Negative Ideal Solution (NIS):")
st.dataframe(pd.DataFrame([negative_ideal_solution], columns=criteria))

# Euclidean distances
distance_pis = np.sqrt(((weighted_matrix - positive_ideal_solution)**2).sum(axis=1))
distance_nis = np.sqrt(((weighted_matrix - negative_ideal_solution)**2).sum(axis=1))
relative_closeness = distance_nis / (distance_pis + distance_nis)

# Combine Step 3 table
step3_table = pd.DataFrame({
    "Alternative": alternatives,
    "Distance from PIS": distance_pis,
    "Distance from NIS": distance_nis,
    "Relative Closeness": relative_closeness
})
st.dataframe(step3_table)

# ---------------- STEP 4: Final Rankings ----------------
st.subheader("Step 4: Final Rankings")
ranking = step3_table.sort_values(by="Relative Closeness", ascending=False).reset_index(drop=True)

# Highlight best alternative
best_stock = ranking.iloc[0]['Alternative']
st.success(f"🏆 The Best Alternative is: {best_stock} 🎉 ✅")

# Display final ranking table
st.dataframe(ranking.style.apply(
    lambda row: ['background-color: lightgreen'] * len(row) if row.name == 0 else ['']*len(row), axis=1))

# ---------------- BAR CHART ----------------
st.subheader("Ranking the Chart")
fig, ax = plt.subplots(figsize=(20,6))  # Wide enough for 100+ alternatives
ax.bar(ranking['Alternative'], ranking['Relative Closeness'], color='skyblue')
ax.set_xlabel("Alternatives")
ax.set_ylabel("Relative Closeness")
ax.set_title("Stock Ranking Using MOORA")
plt.xticks(rotation=90)
st.pyplot(fig)

# ---------------- DOWNLOAD CSV ----------------
st.subheader("Download Result")
csv = ranking.to_csv(index=False).encode('utf-8')
st.download_button("Download Results as CSV", csv, "edurank_results.csv", "text/csv")
