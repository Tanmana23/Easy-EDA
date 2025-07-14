import streamlit as st
import pandas as pd
import numpy as np
import requests
import os

st.set_page_config(page_title="Agentic Data Audit Bot", layout="wide")
st.title("Agentic Data Audit Bot")

# --- Perplexity API Call Function ---
def get_perplexity_recommendation(audit_summary):
    api_key = os.getenv("PPLX_API_KEY")
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
    "model": "sonar-pro",  # or another supported model
    "messages": [
        {"role": "system", "content": "You are a helpful data science assistant."},
        {"role": "user", "content": (
            "Given the following data audit summary, provide a concise, actionable set of preprocessing recommendations. "
            "Use bullet points. Do not exceed 10 sentences. Ensure the output is complete and does not end mid-sentence.\n\n"
            f"{audit_summary}"
        )}
    ],
    "max_tokens": 400,  # Adjust as needed
    "temperature": 0
}

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error from Perplexity API: {response.status_code} - {response.text}"

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("**Preview of Data:**")
    st.dataframe(df.head())

    # --- Run Data Audit ---
    if st.button("Run Data Audit"):
        st.session_state['audit_run'] = True
        st.session_state['df'] = df.copy()
        st.session_state['missing'] = df.isnull().sum()
        st.session_state['duplicates'] = df.duplicated().sum()
        st.session_state['desc'] = df.describe()
        # Collinearity
        corr = df.corr(numeric_only=True).abs()
        high_corr = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                        .stack()
                        .reset_index())
        high_corr.columns = ['Column 1', 'Column 2', 'Correlation']
        st.session_state['high_corr'] = high_corr[high_corr['Correlation'] > 0.8]
        # Outliers
        numeric_cols = df.select_dtypes(include=[np.number])
        z_scores = ((numeric_cols - numeric_cols.mean()) / numeric_cols.std()).abs()
        st.session_state['outlier_counts'] = (z_scores > 3).sum()

    # --- Show Audit Results ---
    if st.session_state.get('audit_run', False):
        st.subheader("Audit Results")
        st.write("**Missing Values per Column:**")
        st.write(st.session_state['missing'])
        st.write(f"**Number of Duplicate Rows:** {st.session_state['duplicates']}")
        st.write("**Basic Statistics:**")
        st.write(st.session_state['desc'])
        st.write("**Highly Correlated Columns (|corr| > 0.8):**")
        if not st.session_state['high_corr'].empty:
            st.dataframe(st.session_state['high_corr'])
        else:
            st.write("No highly correlated column pairs found.")
        st.write("**Potential Outliers (z-score > 3):**")
        st.write(st.session_state['outlier_counts'])

        # --- LLM Recommendations using Perplexity ---
        if st.button("Get Recommendations"):
            audit_summary = f"""Missing values: {st.session_state['missing'].to_dict()}
Duplicates: {st.session_state['duplicates']}
High correlations: {st.session_state['high_corr'].to_dict(orient='records')}
Outliers: {st.session_state['outlier_counts'].to_dict()}"""
            with st.spinner("Generating recommendations..."):
                recommendation = get_perplexity_recommendation(audit_summary)
                st.markdown("**LLM Recommendations:**")
                st.write(recommendation)
        # --- Target Leakage Detection ---
        st.subheader("Target Leakage Detection")
        numeric_columns = st.session_state['df'].select_dtypes(include=[np.number]).columns.tolist()
        target = st.selectbox("Select target column (for leakage check):", ["None"] + numeric_columns)
        if target != "None":
            st.session_state['target_column'] = target
            corr_with_target = st.session_state['df'].corr(numeric_only=True)[target].abs().sort_values(ascending=False)
            st.write("**Feature-Target Correlations:**")
            st.write(corr_with_target)
            st.write("Features with correlation > 0.8 may indicate potential leakage.")

        # --- Data Repair ---
        st.subheader("Apply Data Repairs")
        drop_duplicates = st.checkbox("Drop duplicate rows?", value=False)
        missing_cols = st.session_state['missing'][st.session_state['missing'] > 0].index.tolist()
        repair_options = {}
        if missing_cols:
            st.write("Choose how to repair missing values:")
            for col in missing_cols:
                dtype = str(df[col].dtype)
                if dtype.startswith('float') or dtype.startswith('int'):
                    option = st.selectbox(
                        f"{col} ({dtype})",
                        ["Do nothing", "Impute with mean", "Impute with median", "Drop rows with missing"],
                        key=f"repair_{col}"
                    )
                else:
                    option = st.selectbox(
                        f"{col} ({dtype})",
                        ["Do nothing", "Impute with mode", "Drop rows with missing"],
                        key=f"repair_{col}"
                    )
                repair_options[col] = option

        if st.button("Apply Repairs"):
            df_repaired = st.session_state['df'].copy()
            # Drop duplicates if selected
            if drop_duplicates:
                df_repaired = df_repaired.drop_duplicates()
            # Apply missing value repairs
            for col, action in repair_options.items():
                if action == "Impute with mean":
                    df_repaired[col] = df_repaired[col].fillna(df_repaired[col].mean())
                elif action == "Impute with median":
                    df_repaired[col] = df_repaired[col].fillna(df_repaired[col].median())
                elif action == "Impute with mode":
                    df_repaired[col] = df_repaired[col].fillna(df_repaired[col].mode().iloc[0])
                elif action == "Drop rows with missing":
                    df_repaired = df_repaired[df_repaired[col].notnull()]
                # "Do nothing" does nothing

            st.write("**Cleaned Data Preview:**")
            st.dataframe(df_repaired.head())
            # Download button
            csv = df_repaired.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Cleaned CSV",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )
