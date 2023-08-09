from pathlib import Path

RANDOM_STATE = 8

PATH_X_TRAIN = Path.cwd() / 'input/x_train.npy'
PATH_Y_TRAIN = Path.cwd() / 'input/y_train.npy'

PATH_X_TEST = Path.cwd() / 'input/x_test.npy'
PATH_Y_TEST = Path.cwd() / 'input/y_test.npy'

PATH_MODEL_SELECTION_RES = Path.cwd() / 'output/model_selection_result.csv'

PATH_MODEL = Path.cwd() / 'output/model.pickle'
PATH_PIPELINE = Path.cwd() / 'output/pipeline.pickle'