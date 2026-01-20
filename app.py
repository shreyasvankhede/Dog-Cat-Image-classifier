import streamlit as st
import pandas as pd
import numpy as np
from models import load_model,preprocess_input,train_model


st.title("Dogs vs Cats Image classifier")


model=load_model("cat_dog_classifier_transfer.h5")
if model==-1:
    st.warning("Model not found,training new model")
    model=train_model()


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = preprocess_input(uploaded_file)

    st.image(uploaded_file, caption="Uploaded Image")

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        confidence = prediction * 100
        st.success(f"Prediction: Dog 🐶 ({confidence:.2f}%)")
    else:
        confidence = (1 - prediction) * 100
        st.success(f"Prediction: Cat 🐱 ({confidence:.2f}%)")

    