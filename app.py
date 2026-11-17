
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("genetic_disorder_model.pkl")

st.set_page_config(
    page_title="Genetic Disorder Risk Classification",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Genetic Disorder Risk Classification")
st.write("Predict the genetic disorder category using Machine Learning.")

st.warning(
    "This application is an educational prototype "
    "and not a medical diagnosis tool."
)

st.header("Patient Information")

# Input fields
patient_age = st.number_input(
    "Patient Age", min_value=0, max_value=100, value=5
)

mother_age = st.number_input(
    "Mother's Age", min_value=10, max_value=100, value=25
)

father_age = st.number_input(
    "Father's Age", min_value=10, max_value=100, value=30
)

gender = st.selectbox(
    "Gender", ["Male", "Female", "Ambiguous"]
)

genes_mother = st.selectbox(
    "Genes in mother's side", ["Yes", "No"]
)

inherited_father = st.selectbox(
    "Inherited from father", ["Yes", "No"]
)

maternal_gene = st.selectbox(
    "Maternal gene", ["Yes", "No"]
)

paternal_gene = st.selectbox(
    "Paternal gene", ["Yes", "No"]
)

st.info(
    "The complete model input fields and their dataset values "
    "must be provided before prediction can work."
)

st.write("Model loaded successfully:", type(model).__name__)
