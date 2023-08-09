import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import predict

# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle")
)

stroke_width = st.sidebar.slider("Stroke width: ", 20, 60, 20)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")

realtime_update = True

# create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)", 
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    update_streamlit=realtime_update,
    height=784,
    width=784, 
    drawing_mode=drawing_mode,
    key="canvas",
)

# predict here
img_cmyk= canvas_result.image_data
if img_cmyk is not None:

    # convert to grayscale
    img_gray = np.sum(img_cmyk, axis=-1)
    # rescale the pixel values to be in the range [0, 255]
    num = (img_gray - np.min(img_gray))
    den = (np.max(img_gray) - np.min(img_gray))
    img_gray = num // den * 255

    # predict
    y_pred = predict.predict(x=img_gray, with_preprocesing= True)
    st.write(y_pred.shape)
    print(y_pred)

    