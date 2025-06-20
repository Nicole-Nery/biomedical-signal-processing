import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from functions import *
from utils_display import *

def app():
    with st.sidebar:
        st.title("Signal Generator")
        signal_type = st.selectbox("Waveform", ["Sinusoidal", "Square", "Triangular", "Random"])

        with st.expander("Basic parameters"):
            fs = st.slider("Sample rate (Hz)", 10, 1000, 100, step=1)
            duration = st.slider("Signal duration (s)", 1, 10, 5, step=1)
            offset = st.slider("Offset", -10.0, 10.0, 0.0, step=0.5)
            amplitude = st.slider("Amplitude", 0.1, 10.0, 1.0, step=0.1)
            frequency = st.slider("Frequency (Hz)", 0.1, 50.0, 1.0, step=0.1)

        df = None
        if signal_type == "Sinusoidal":
            df = generateSine(offset, amplitude, fs, duration, frequency)
        elif signal_type == "Square":
            df = generateSquare(offset, amplitude, fs, duration, frequency)
        elif signal_type == "Triangular":
            df = generateTriangle(offset, amplitude, fs, duration, frequency)
        elif signal_type == "Random":
            distribution = st.selectbox("Distribution", ["normal", "uniform", "binomial"])
            df = generateRandomSignal(offset, amplitude, fs, duration, distribution)

        with st.expander("Additional effects"):
            add_noise = st.checkbox("Add noise")
            add_trend = st.checkbox("Add trend")
            add_discontinuity = st.checkbox("Add discontinuity")
            add_sudden_change = st.checkbox("Sudden amplitude change")

            if add_noise:
                snr = st.slider("SNR (dB)", 0, 50, 20, step=1)
                df = addNoise(df, snr)

            if add_trend:
                trend_type = st.selectbox("Trend type", ["linear", "quadratic", "sinusoidal"])

                a = b = c = offset_trend = amp_trend = f_trend = 0

                if trend_type == 'linear':
                    a = st.number_input("Coefficient a", key='a_linear', step=0.1)
                    b = st.number_input("Coefficient b", key='b_linear', step=0.1)

                elif trend_type == 'quadratic':
                    a = st.number_input("Coefficient a", key='a_quad', step=0.1)
                    b = st.number_input("Coefficient b", key='b_quad', step=0.1)
                    c = st.number_input("Coefficient c", key='c_quad', step=0.1)

                elif trend_type == 'sinusoidal':
                    offset_trend = st.slider("Offset", -10.0, 10.0, 0.0, step=0.5, key='offset_trend')
                    amp_trend = st.slider("Amplitude", 0.1, 10.0, 1.0, step=0.1, key="amplitude_trend")
                    f_trend = st.slider("Frequency (Hz)", 0.1, 50.0, 1.0, step=0.1, key="f_trend")

                df = addTrend(df, trend_type, a, b, c, offset_trend, amp_trend, f_trend)

            if add_discontinuity:
                break_time = st.slider("Discontinuity time (s)", 0.0, float(duration), 2.0)
                jump_value = st.slider("Jump value", -10.0, 10.0, 2.0)
                df = addDiscontinuity(df, break_time, jump_value)

            if add_sudden_change:
                change_time = st.slider("Time of change (s)", 0.0, float(duration), 2.0)
                new_amp = st.slider("New amplitude", 0.1, 10.0, 2.0, step=0.1)
                df = addSuddenChange(df, change_time, new_amp)

    if df is not None:
        display_signal_and_features(df)

app()
