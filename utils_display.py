import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from functions import *

def display_signal_and_features(df):
    with st.sidebar:
        with st.container(border=True):
            window_size_sec = st.number_input("Window Size (seconds)", min_value=0.1, max_value=float(df['t'].iloc[-1]), value=0.1, step=0.1)
            overlap_percent = st.number_input('Overlap percent', min_value=0.0, max_value=0.9, value=0.0, step=0.05)
            df_features = extract_features_windowed(df, window_size_sec, overlap_percent)

    with st.expander("Features per Window", icon=":material/table_view:"):
        st.markdown("#### Features per Window")
        st.dataframe(df_features, height=220, hide_index=True)

    with st.container():
        col1, col2 = st.columns([1,1])

        with col1:
            with st.container(border=True, height=260):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['t'], 
                    y=df['signal'], 
                    mode='lines', 
                    name='Signal',
                    line=dict(color="#3a19e4")
                    ))
                fig.update_layout(
                    xaxis_title="Tempo (s)",
                    yaxis_title="Amplitude",
                    template="plotly_white",
                    height=290
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            with st.container(border=True, height=260):

                fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))

                im1 = axes[0].imshow(applyGAF(df, 'summation'), cmap='hot', origin='lower')
                axes[0].set_title('Gramian Angular Summation Field', fontsize=7, pad=10)
                cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
                cbar1.ax.tick_params(labelsize=6)

                im2 = axes[1].imshow(applyGAF(df, 'difference'), cmap='hot', origin='lower')
                axes[1].set_title('Gramian Angular Difference Field', fontsize=7, pad=10)
                cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
                cbar2.ax.tick_params(labelsize=6)

                axes[0].tick_params(axis='both', labelsize=6)
                axes[1].tick_params(axis='both', labelsize=6)

                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

    with st.container(border=True, height=320):
        st.markdown("#### Feature Trends Across Time Windows")

        tabs = st.tabs(["Plots", "Definitions"])
        with tabs[0]:
            features_to_plot = [col for col in df_features.columns if col not in ['Window', 'Mean_Time']]

            for idx, feature in enumerate(features_to_plot):
                with st.container(border=False):
                    col1, col2 = st.columns([1,1])
                    with col1:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df_features['Window'], 
                            y=df_features[feature], 
                            mode='lines', 
                            name=feature,
                            line=dict(color="#3a19e4")
                        ))
                        fig.update_layout(
                            title=f"Feature Variation: {feature}",
                            xaxis_title="Window",
                            yaxis_title=feature,
                            template="plotly_white",
                            height=300,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        fig, axes = plt.subplots(1, 2, figsize=(6, 4))
                        
                        df = pd.DataFrame({'signal': df_features[feature].values})

                        im1 = axes[0].imshow(applyGAF(df, 'summation'), cmap='hot', origin='lower')
                        axes[0].set_title('Gramian Angular Summation Field', fontsize=8, pad=10)
                        cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
                        cbar1.ax.tick_params(labelsize=7)

                        im2 = axes[1].imshow(applyGAF(df, 'difference'), cmap='hot', origin='lower')
                        axes[1].set_title('Gramian Angular Difference Field', fontsize=8, pad=10)
                        cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
                        cbar2.ax.tick_params(labelsize=7)

                        axes[0].tick_params(axis='both', labelsize=7)
                        axes[1].tick_params(axis='both', labelsize=7)

                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)

        with tabs[1]:
            st.markdown("""
                - **MAV (Mean Absolute Value)**  
                It quantifies the average magnitude of the signal by calculating the mean of the absolute values of the amplitude of the signal over a specific time window
                
                - **MAVFD (Mean Absolute Value of First Difference)**  
                The mean absolute value of the first-order difference of the signal.

                - **MAVSD (Mean Absolute Value of Second Difference)**  
                The mean absolute value of the second-order difference of the signal.

                - **RMS (Root Mean Square)**  
                The square root of the mean of the squares of the signal samples.

                - **PEAK_VALUE (Peak Amplitude)**  
                The maximum absolute value in the signal.

                - **ZERO_CROSSINGS**  
                It represents the rate at which a signal crosses the zero-axis over a given period of time.

                - **MEAN_FREQUENCY**  
                The average frequency, weighted by the power spectrum.

                - **PEAK_FREQUENCY**  
                The frequency with the highest power in the spectrum.

                - **F50**  
                The frequency below which 50% of the total spectral power resides.

                - **F80**  
                The frequency below which 80% of the total spectral power resides.

                - **BAND_POWER_3_5_TO_7_5**  
                Total power in the frequency band from 3.5 Hz to 7.5 Hz.

                - **FUZZY_ENTROPY**  
                A measure of signal complexity and unpredictability.

                - **APPROX_ENTROPY**  
                Another entropy-based measure reflecting signal regularity.

                - **VARIANCE**  
                The statistical variance of the signal.

                - **RANGE**  
                Difference between the maximum and minimum values of the signal.

                - **IQR (Interquartile Range)**  
                The range between the 25th and 75th percentiles.

                - **SKEWNESS**  
                A measure of the asymmetry of the signal distribution.

                - **KURTOSIS**  
                A measure of the "tailedness" of the signal distribution.

                - **ACTIVITY**  
                As described by Hjorth, the first parameter, known as activity, represents the average power of the signals.
                Activity provides a measure of the squared deviation of the amplitude.

                - **MOBILITY**  
                The second Hjorth parameter, mobility, indicates the mean frequency of the signal.  
                Mobility provides a measure of the standard deviation of the slope concerning the standard deviation of the amplitude.
                
                - **COMPLEXITY**  
                The third Hjorth parameter, complexity, characterizes the shape of the signal's curve. For instance, the complexity value of a pure sine wave converges to one.
                Complexity gives us the number of standard slopes obtained through the average time required for generation of one standard amplitude.
                        """)

