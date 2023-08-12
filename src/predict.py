import config
import pickle


# predict
def predict(model, x, with_preprocesing=True):
    
    if model == "Random Forest":
        path_model = config.PATH_MODEL_RF
    elif model == "K-Nearest Classifier":
        path_model = config.PATH_MODEL_KNC
    elif model == "Logistic Regression":
        path_model = config.PATH_MODEL_LR
    else:
        raise ValueError("Model unrecognized.")

    # load model and pipeline
    with open(path_model, "rb") as file:
        model = pickle.load(file)
    with open(config.PATH_PIPELINE, "rb") as file:
        pipeline = pickle.load(file)

    # preprocess
    if with_preprocesing:
        x = pipeline.transform(x)
        return model.predict(x)

    # predict
    return model.predict(x)
