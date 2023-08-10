import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from skimage.transform import resize
from PIL import Image

import predict
import utils

# specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle")
)

stroke_width = 40
stroke_color = st.sidebar.color_picker("Stroke color hex: ")

realtime_update = True

st.header("Draw here and I will predict!")

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
img_cmyk = canvas_result.image_data
if img_cmyk is not None:
    pass
    # TODO: convert RGBA to grayscale
    # convert rgba to grayscale 
    # rescale
    # then convert back to array
    # img_gray = Image.fromarray(img_cmyk, "RGBA").convert('LA')
    # img_gray = resize(np.asarray(img_gray), output_shape= (28, 28))
    
    # # flatten
    # img_gray = img_gray.flatten().reshape(1, -1)

    # st.write(img_gray.shape)


    # # predict
    # try:
    #     y_pred = predict.predict(x=img_gray, with_preprocesing=True)
    #     result = f"This is :red[{y_pred[0]}]!"
    #     st.header(result)
    # except ValueError:
    #     pass
