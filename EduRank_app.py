import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# ---------------- APP TITLE ----------------
st.title("EduRank: MOORA-Based Stock Selection for Educational Innovation")
st.markdown("""
This app evaluates and ranks stocks based on multiple criteria using the **MOORA** method.
Upload your stock dataset, assign weights to each criterion, choose benefit/cost criteria,
and view rankings with tables, a bar chart, and download the step-by-step report.
""")

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader("Upload Excel or CSV file with stock data", type=["csv", "xlsx"])

# Example data
def load_example():
    data = {
        'Alternative': ['A', 'B', 'C'],
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

# Ensure first column is Alternative
df = df.rename(columns={df.columns[0]: "Alternative"})

st.subheader("Stock Data")
st.dataframe(df)

alternatives = df["Alternative"]
criteria = df.columns[1:]
data = df.iloc[:, 1:].astype(float)

# ---------------- STEP 0: INPUT WEIGHTS & SELECT CRITERIA ----------------
st.subheader("Step 0: Input Weights and Select Criteria")
weights = [st.number_input(f"Weight for {col}", min_value=0.0, max_value=1.0, 
                           value=1/len(criteria), step=0.01) for col in criteria]

weight_sum = round(sum(weights),4)
st.write(f"**Current weight sum:** {weight_sum}")

# Select benefit and cost criteria
benefit_criteria = st.multiselect("Select Benefit Criteria", criteria.tolist())
cost_criteria = st.multiselect("Select Cost Criteria", [c for c in criteria if c not in benefit_criteria])

if weight_sum != 1:
    st.warning("Weights must sum to 1 to proceed.")
elif not benefit_criteria and not cost_criteria:
    st.warning("Please select at least one benefit or cost criterion.")
else:
    # ---------------- STEP 1: NORMALIZE ----------------
    st.subheader("Step 1: Normalize the Data")
    normalized_matrix = data.copy()
    for i, col in enumerate(criteria):
        normalized_matrix[col] = data[col] / np.sqrt((data[col]**2).sum())
    normalized_df = pd.concat([alternatives, normalized_matrix], axis=1)
    st.dataframe(normalized_df)

    # ---------------- STEP 2: WEIGHTED NORMALIZED ----------------
    st.subheader("Step 2: Weighted Normalized Matrix")
    weighted_matrix = normalized_matrix.copy()
    for i, col in enumerate(criteria):
        weighted_matrix[col] = normalized_matrix[col] * weights[i]
    weighted_df = pd.concat([alternatives, weighted_matrix], axis=1)
    st.dataframe(weighted_df)

    # ---------------- STEP 3: PERFORMANCE SCORE ----------------
    st.subheader("Step 3: Calculate Performance Score (Benefit − Cost)")

    benefit_data = weighted_matrix[benefit_criteria] if benefit_criteria else pd.DataFrame(np.zeros((len(alternatives),0)))
    cost_data = weighted_matrix[cost_criteria] if cost_criteria else pd.DataFrame(np.zeros((len(alternatives),0)))

    performance_score = benefit_data.sum(axis=1) - cost_data.sum(axis=1)

    perf_df = pd.DataFrame({
        "Alternative": alternatives,
        "Performance Score": performance_score.round(4)
    })
    st.dataframe(perf_df)

    # ---------------- STEP 4: FINAL RANKINGS ----------------
    st.subheader("Step 4: Final Rankings")
    ranking = perf_df.sort_values('Performance Score', ascending=False).reset_index(drop=True)
    ranking['Rank'] = range(1, len(ranking) + 1)

    # Format to 4 decimals for display
    ranking['Performance Score'] = ranking['Performance Score'].map('{:.4f}'.format)

    # Highlight the top-ranked alternative in green
    def highlight_top(row):
        return ['background-color: lightgreen'] * len(row) if row.name == 0 else [''] * len(row)

    st.dataframe(ranking.style.apply(highlight_top, axis=1))

    # Announce the best alternative
    best_alt = ranking.loc[0, 'Alternative']
    st.success(f"🏆 **The Best Alternative is:** {best_alt} 🎉💹")

    # ---------------- STEP 5: BAR CHART ----------------
    st.subheader("Step 5: Visualize Performance Scores")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(ranking['Alternative'], ranking['Performance Score'].astype(float), color='skyblue')
    ax.set_xlabel("Alternatives")
    ax.set_ylabel("Performance Score")
    ax.set_title("Stock Ranking Using MOORA (Performance Score)")
    plt.xticks(rotation=0)
    st.pyplot(fig)

    # ---------------- DOWNLOAD EXCEL REPORT ----------------
    st.subheader("Download Step-by-Step Excel Report")
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        normalized_df.to_excel(writer, index=False, sheet_name='Step 1 - Normalized')
        weighted_df.to_excel(writer, index=False, sheet_name='Step 2 - Weighted Normalized')
        perf_df.to_excel(writer, index=False, sheet_name='Step 3 - Performance Score')
        ranking.to_excel(writer, index=False, sheet_name='Step 4 - Final Ranking')

    excel_data = output.getvalue()

    st.download_button(
        label="📥 Download Full Step-by-Step Excel Report",
        data=excel_data,
        file_name="EduRank_Full_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
