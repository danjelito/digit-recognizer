import streamlit as st
from streamlit_drawable_canvas import st_canvas
from skimage.transform import resize_local_mean
from skimage.color import rgb2gray
import numpy as np
from PIL import Image

import predict

# specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle")
)

stroke_width = 40
stroke_color = "black"

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
img_rgba = canvas_result.image_data
if img_rgba is not None:
    
    # get alpha channel only
    # convert to PIL image
    img_gray = Image.fromarray(img_rgba[:, :, -1])
    # resize 
    img_gray = img_gray.resize((28, 28))
    # convert back to array
    # flatten
    img_flat = np.asarray(img_gray).reshape((1, 784))

# predict if there is image
if np.sum(img_rgba) != 0:
    try:
        y_pred = predict.predict(x=img_flat, with_preprocesing=True)
        result = f"This is :red[{y_pred[0]}]!"
        st.header(result)
    except ValueError:
        pass

    # st.write(np.max(img_gray))
    # st.write(np.shape(img_gray))
    # st.image(img_gray)