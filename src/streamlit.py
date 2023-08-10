import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

import predict
import utils

# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle")
)

stroke_width = 40
stroke_color = st.sidebar.color_picker("Stroke color hex: ")

realtime_update = True

st.header('Draw here and I will predict!')

# create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)", 
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    update_streamlit=realtime_update,
    height=500,
    width=500, 
    drawing_mode=drawing_mode,
    key="canvas",
)

# predict here
img_cmyk= canvas_result.image_data
if img_cmyk is not None:

    # convert to grayscale
    img_gray = np.sum(img_cmyk, axis=-1)
    # rescale the pixel values to be in the range [0, 255]
    img_gray = utils.rescale_array(img_gray, (0, 255))
    # resize the image to 28x28
    img_gray = resize(img_gray, (28, 28))
    # flatten
    img_gray = img_gray.flatten().reshape(1, -1)

    # predict
    try:
        y_pred = predict.predict(x=img_gray, with_preprocesing= True)
        result = f'This is :red[{y_pred[0]}]!'
        st.header(result)
    except ValueError:
        pass

    