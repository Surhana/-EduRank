    # ---------------- STEP 3: MOORA Score (Benefit - Cost) ----------------
    st.subheader("Step 3: Calculate MOORA Score (Benefit − Cost)")

    # Ask user to classify criteria
    benefit_criteria = st.multiselect("Select Benefit Criteria Columns", criteria.tolist())
    cost_criteria = st.multiselect("Select Cost Criteria Columns", [c for c in criteria if c not in benefit_criteria])

    # Compute MOORA scores
    benefit_data = weighted_matrix[benefit_criteria] if benefit_criteria else pd.DataFrame(np.zeros((len(alternatives),0)))
    cost_data = weighted_matrix[cost_criteria] if cost_criteria else pd.DataFrame(np.zeros((len(alternatives),0)))

    moora_score = benefit_data.sum(axis=1) - cost_data.sum(axis=1)

    moora_df = pd.DataFrame({
        "Alternative": alternatives,
        "MOORA Score (Benefit-Cost)": moora_score.round(4)
    })
    st.dataframe(moora_df)

    # ---------------- STEP 4: Final Rankings ----------------
    st.subheader("Step 4: Final Rankings")
    ranking = moora_df.sort_values('MOORA Score (Benefit-Cost)', ascending=False).reset_index(drop=True)
    ranking['Rank'] = range(1, len(ranking) + 1)

    # Highlight the top-ranked alternative in green
    def highlight_top(row):
        return ['background-color: lightgreen'] * len(row) if row.name == 0 else [''] * len(row)

    st.dataframe(ranking.style.apply(highlight_top, axis=1))

    # Announce the best alternative
    best_alt = ranking.loc[0, 'Alternative']
    st.success(f"🏆 **The Best Alternative is:** {best_alt} 🎉💹")
