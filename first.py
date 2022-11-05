import streamlit as st
from PIL import Image

st.title('Uzumaki Nagato')

image = Image.open("pain.jpg")

st.image(image, use_column_width = True)

st.subheader('Know pain, feel pain, accept pain')

