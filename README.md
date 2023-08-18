# Digit Recognition with Machine Learning and Streamlit

## Overview
This project demonstrates the integration of machine learning and user-friendly interfaces using Streamlit. The application allows users to draw digits on a canvas, and machine learning models predict the corresponding digit. It's a simple yet effective way to showcase the power of machine learning in image recognition tasks.

## Features
- Choose from a variety of machine learning models, including Multi-layer Perceptron, K-Nearest Classifier, Random Forest, and more.
- Experiment with different drawing tools like Freedraw, Line, Rect, or Circle on the digital canvas.
- Witness real-time predictions as the machine learning model processes your input.

## Model Description
The models employed in this project are deliberately simple to highlight their effectiveness despite their straightforward architecture. You can learn more about each model's strengths and weaknesses in the `model_description.py` file.

## Deployment
The application is deployed and accessible at: [https://anj-digit-recognizer.streamlit.app/](https://anj-digit-recognizer.streamlit.app/)

## Project Structure
```
|-- output
| |-- model # Directory for saved models
| |-- pipeline.pickle # Pipeline artifacts
| |-- model_selection_result.csv # Model selection results
|
|-- src
| |-- config.py # Configuration settings for the project
| |-- dataset.py # Data loading and preprocessing functions
| |-- hyperparam.py # Hyperparameter configurations
| |-- model_description.py # Descriptions of models' strengths and weaknesses
| |-- model.py # Implementation of machine learning models
| |-- model_selection.py # Model selection process and evaluation
| |-- predict.py # Prediction functions
| |-- requirements.txt # List of project dependencies
| |-- sandbox.py # Experimental code snippets
| |-- streamlit.py # Streamlit application code
| |-- train_test.py # Training and testing of models
| |-- utils.py # Utility functions
|
|-- README.md
```

## Contact
If you'd like to connect or learn more about this project, you can find me on LinkedIn: [Devan Anjelito](https://www.linkedin.com/in/devan-anjelito/).