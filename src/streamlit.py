import streamlit as st
from streamlit_drawable_canvas import st_canvas
from skimage.transform import resize_local_mean
from skimage.color import rgb2gray
import numpy as np
from PIL import Image

import predict


with st.sidebar:
    st.header("Digit Recognizer")
    st.markdown(":gray[_by Devan Anjelito_]")
    st.markdown("\n")
    st.markdown(
        """This is a demonstration of how a machine learning model
        can "see" and predict handwriting.
        """
    )
    st.markdown("\n")
    st.markdown(
        """The model used are all simple model which, extraordinarily,
        works quite well, given the simple architecture of the model.
        """
    )
    st.markdown("\n")
    st.markdown(
        """If you are interested to know more,
        check out my other projects on [my website](https://danjelito.github.io/)
        or greet me via [LinkedIn](https://www.linkedin.com/in/devan-anjelito/).
        """
    )
    st.divider()

with st.sidebar:
    model = st.selectbox(
        "Select ML model:",
        (
            "Multi-layer Perceptron",
            "K-Nearest Classifier",
            "Random Forest",
            "Logistic Regression",
            "Quadratic Discriminant Analysis",
            "Decision Tree",
        ),
    )

    drawing_mode = st.selectbox("Drawing tool:", ("freedraw", "line", "rect", "circle"))


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
            with st.spinner("Wait for it, some models are slower than others..."):
                y_pred = predict.predict(model=model, x=img_flat, with_preprocesing=True)
            result = f"This is :red[{y_pred[0]}]!"
            st.header(result)
        except ValueError:
            pass
