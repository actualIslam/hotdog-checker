import streamlit as st
from PIL import Image
from transformers import pipeline

pipe = pipeline("image-classification", model="julien-c/hotdog-not-hotdog")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    pred = pipe(image)

    st.write(pred)
    
    st.image(uploaded_file)