import config
import pickle
import model

# load model and pipeline
with open(config.PATH_MODEL, "rb") as file:
    model = pickle.load(file)
with open(config.PATH_PIPELINE, "rb") as file:
    pipeline = pickle.load(file)

# predict
def predict(x, with_preprocesing= True):
    
    if with_preprocesing:
        x = pipeline.transform(x)
        return model.predict(x)
    
    return model.predict(x)