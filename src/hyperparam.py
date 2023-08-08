import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import optuna

import model
import config
import utils

def main(n_trials):

    # load dataset
    x_train, y_train = utils.load_dataset(load_test_set= False)

    # preprocessing
    model.pipeline.fit(x_train, y_train)
    x_train = model.pipeline.transform(x_train)

    def objective(trial):
    
        # param space
        param = {
            'n_neighbors': trial.suggest_int('n_neighbors', 1, 7), 
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
            'leaf_size': trial.suggest_int('leaf_size', 10, 60), 
            'n_jobs': 1,
        }    
        clf= KNeighborsClassifier(**param)
        f1= cross_val_score(clf, x_train, y_train, scoring= 'f1_weighted', cv= 3, n_jobs= -1, verbose= 0)
        return np.mean(f1)

    study= optuna.create_study(direction= 'maximize')
    study.optimize(objective, n_trials= n_trials)

if __name__ == "__main__":
    main(n_trials= 100)