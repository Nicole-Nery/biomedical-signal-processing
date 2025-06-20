import streamlit as st

# Configurações visuais
st.set_page_config(
    page_title="BioSignal",
    layout="wide"
)

# Estilo CSS
with open("main.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Navegação entre páginas
pages = [
    st.Page("Synthetic_Signal.py", title="Generate Synthetic Signal", icon=":material/airware:"),
    st.Page("Upload_Signal.py", title="Upload Signal", icon=":material/earthquake:"),
]

page = st.navigation(pages)
page.run()
