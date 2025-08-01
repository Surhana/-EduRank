import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title("EduRank: MOORA-Based Stock Selection for Educational Innovation")
st.markdown("""
This app evaluates and ranks stocks based on multiple criteria using the **MOORA (Multi-Objective Optimization on the Basis of Ratio Analysis)** method.
Upload your stock dataset, assign weights to each criterion, and view rankings with a clear bar chart.
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

st.subheader("Stock Data")
st.dataframe(df)

stocks = df.iloc[:, 0]
criteria = df.columns[1:]
data = df.iloc[:, 1:].astype(float)

# Input weights
st.subheader("Input Weights (must sum to 1)")
weights = [st.number_input(f"Weight for {col}", min_value=0.0, max_value=1.0, value=1/len(criteria), step=0.01)
           for col in criteria]

if sum(weights) != 1:
    st.warning("Weights must sum to 1! Please adjust the weights.")

# Step 1: Normalize
st.subheader("Step 1: Normalize the Data")
normalized = data.copy()
for i, col in enumerate(criteria):
    normalized[col] = data[col] / np.sqrt((data[col]**2).sum())
st.dataframe(normalized)

# Step 2: Weighted Normalized Matrix
st.subheader("Step 2: Weighted Normalized Matrix")
weighted = normalized.copy()
for i, col in enumerate(criteria):
    weighted[col] = weighted[col] * weights[i]
st.dataframe(weighted)

# Step 3: MOORA Performance Index (PIS) with values
st.subheader("Step 3: MOORA Performance Index (PIS)")

# Calculate PIS and NIS
positive_ideal_solution = weighted.max()
negative_ideal_solution = weighted.min()

st.write("Positive Ideal Solution (PIS):")
st.dataframe(pd.DataFrame(positive_ideal_solution).T)
st.write("Negative Ideal Solution (NIS):")
st.dataframe(pd.DataFrame(negative_ideal_solution).T)

# Calculate Euclidean distances
distance_pis = np.sqrt(((weighted - positive_ideal_solution)**2).sum(axis=1))
distance_nis = np.sqrt(((weighted - negative_ideal_solution)**2).sum(axis=1))

# Show distances
st.write("Euclidean Distance from PIS:")
st.dataframe(pd.DataFrame(distance_pis, columns=["Distance from PIS"]))
st.write("Euclidean Distance from NIS:")
st.dataframe(pd.DataFrame(distance_nis, columns=["Distance from NIS"]))

# Relative closeness
relative_closeness = distance_nis / (distance_pis + distance_nis)
st.write("Relative Closeness Scores:")
st.dataframe(pd.DataFrame(relative_closeness, columns=["Relative Closeness"]))

# Step 4: Final Rankings
st.subheader("Step 4: Final Rankings")
ranking = pd.DataFrame({
    'Stock': stocks,
    'Distance from PIS': distance_pis,
    'Distance from NIS': distance_nis,
    'Relative Closeness': relative_closeness
}).sort_values(by="Relative Closeness", ascending=False).reset_index(drop=True)

# Highlight best alternative
best_stock = ranking.iloc[0]['Stock']
st.success(f"🏆 The Best Alternative is: {best_stock} 🎉 ✅")

# Display final ranking
st.dataframe(ranking.style.apply(lambda row: ['background-color: lightgreen'] * len(row) if row.name == 0 else ['']*len(row), axis=1))

# Bar Chart for Ranking
st.subheader("Ranking the Chart")
fig, ax = plt.subplots(figsize=(20,6))  # Wide figure for 100+ stocks
ax.bar(ranking['Stock'], ranking['Relative Closeness'], color='skyblue')
ax.set_xlabel("Stocks")
ax.set_ylabel("Relative Closeness")
ax.set_title("Stock Ranking Using MOORA")
plt.xticks(rotation=90)  # Rotate labels for readability
st.pyplot(fig)

# Download Results
st.subheader("Download Result")
csv = ranking.to_csv(index=False).encode('utf-8')
st.download_button("Download Results as CSV", csv, "edurank_results.csv", "text/csv")
