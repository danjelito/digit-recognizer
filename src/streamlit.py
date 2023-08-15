import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image

import predict
from model_description import model_descriptions, model_strengths, model_weaknesses

# set page configuration
st.set_page_config(
    page_title= 'Digit Recognizer',
    page_icon= '🔢',
    initial_sidebar_state= 'expanded',
    layout = 'wide'
)

# opening text
with st.sidebar:
    st.title("Digit Recognizer")
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

# model selection
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
    drawing_mode = st.selectbox("Drawing tool:", ("Freedraw", "Line", "Rect", "Circle"))

# set columns
col1, col2 = st.columns([0.3, 0.7], gap='large')

# model description
with col1:

    text_header = f'''You are using :blue[{model}] model!'''
    st.subheader(text_header)
    st.markdown(model_descriptions[model])
    st.markdown(f'**Strength** : {model_strengths[model]}')
    st.markdown(f'**Weakness** : {model_weaknesses[model]}')

# canvas and result
with col2:

    st.subheader("Draw a number here and I will predict!")

    # create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=40,
        stroke_color='black',
        update_streamlit=True,
        height=500,
        width=500,
        drawing_mode=drawing_mode.lower(),
        key="canvas",
    )

    # prediction
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
