import streamlit as st
import pandas as pd
from functions import *
from utils_display import *

def app():
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload your signal file (.csv or .txt)", type=["csv", "txt"])
        df = None

        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df_uploaded = pd.read_csv(uploaded_file)
                    df_uploaded = df_uploaded.loc[:, ~df_uploaded.columns.str.contains('^Unnamed')]
                else:
                    df_uploaded = pd.read_csv(uploaded_file, delimiter="\t")

                df_uploaded.columns = [str(c).strip() for c in df_uploaded.columns]

                st.success("File loaded successfully!")

                col_time = st.selectbox("Select the time column", options=df_uploaded.columns)
                col_signal = st.selectbox("Select the signal column", options=df_uploaded.columns)

                if col_time == col_signal:
                    st.warning("Time and signal columns must be different.")
                else:
                    df = pd.DataFrame({
                        "t": df_uploaded[col_time],
                        "signal": df_uploaded[col_signal]
                    })

            except Exception as e:
                st.error(f"Error loading file: {e}")

    if df is not None:
        display_signal_and_features(df)

app()
