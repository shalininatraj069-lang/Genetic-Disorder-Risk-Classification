
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Genetic Disorder Risk Classification",
    page_icon="🧬",
    layout="wide"
)

@st.cache_resource
def load_files():
    model = joblib.load("genetic_disorder_model.pkl")
    metadata = joblib.load("model_metadata.pkl")
    return model, metadata

try:
    model, metadata = load_files()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

st.title("🧬 Genetic Disorder Risk Classification")
st.write("Machine Learning Based Genetic Disorder Category Prediction")

st.warning(
    "Educational prototype only. This application "
    "does not provide medical diagnosis."
)

st.header("Enter Patient Information")

input_data = {}

with st.form("patient_form"):

    for column, details in metadata.items():

        if details["type"] == "numeric":

            input_data[column] = st.number_input(
                column,
                value=float(details["default"])
            )

        else:

            options = details["options"]

            if options:
                input_data[column] = st.selectbox(
                    column,
                    options,
                    index=options.index(details["default"])
                    if details["default"] in options else 0
                )
            else:
                input_data[column] = ""

    submitted = st.form_submit_button("Predict Disorder")

if submitted:

    try:
        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)[0]

        st.success("Prediction Completed!")

        st.subheader("Predicted Genetic Disorder")
        st.write(f"### {prediction}")

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_df)[0]

            probability_df = pd.DataFrame({
                "Genetic Disorder": model.classes_,
                "Probability (%)": probabilities * 100
            })

            st.subheader("Prediction Probability")
            st.bar_chart(
                probability_df.set_index("Genetic Disorder")
            )

            st.dataframe(
                probability_df.style.format({
                    "Probability (%)": "{:.2f}%"
                }),
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Prediction Error: {e}")

