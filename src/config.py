from pathlib import Path

RANDOM_STATE = 8

PATH_X_TRAIN = Path.cwd() / 'input/x_train.npy'
PATH_Y_TRAIN = Path.cwd() / 'input/y_train.npy'

PATH_X_TEST = Path.cwd() / 'input/x_test.npy'
PATH_Y_TEST = Path.cwd() / 'input/y_test.npy'

PATH_MODEL_SELECTION_RES = Path.cwd() / 'output/model_selection_result.csv'

PATH_MODEL_MLP = Path.cwd() / 'output/model/model_mlp.pickle'
PATH_MODEL_KNC = Path.cwd() / 'output/model/model_knc.pickle'
PATH_MODEL_RF = Path.cwd() / 'output/model/model_rf.pickle'
PATH_MODEL_DT = Path.cwd() / 'output/model/model_dt.pickle'
PATH_MODEL_LR = Path.cwd() / 'output/model/model_lr.pickle'
PATH_MODEL_QDA = Path.cwd() / 'output/model/model_qda.pickle'

PATH_PIPELINE = Path.cwd() / 'output/pipeline.pickle'