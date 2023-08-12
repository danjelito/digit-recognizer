import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import optuna

import model
import utils


# load dataset
x_train, y_train = utils.load_dataset(load_test_set=False)

# preprocessing
model.pipeline.fit(x_train, y_train)
x_train = model.pipeline.transform(x_train)


# knc
def optimize_knc(n_trials):
    def objective(trial):
        param = {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 7),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "algorithm": trial.suggest_categorical(
                "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
            ),
            "leaf_size": trial.suggest_int("leaf_size", 10, 60),
            "n_jobs": -1,
        }
        clf = KNeighborsClassifier(**param)
        f1 = cross_val_score(
            clf, x_train, y_train, scoring="f1_weighted", cv=3, n_jobs=-1, verbose=0
        )
        return np.mean(f1)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    return study.best_params


# random forest
# set small model for faster performance
def optimize_rf(n_trials):
    def objective(trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 2, 50),
            "criterion": trial.suggest_categorical(
                "criterion", ["gini", "entropy", "log_loss"]
            ),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "n_jobs": -1,
        }
        clf = RandomForestClassifier(**param)
        f1 = cross_val_score(
            clf, x_train, y_train, scoring="f1_weighted", cv=3, n_jobs=-1, verbose=0
        )
        return np.mean(f1)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    return study.best_params


# logreg
def optimize_lr(n_trials):
    def objective(trial):
        param = {
            "C": trial.suggest_float("C", 0.00001, 10, log=True),
            "solver": trial.suggest_categorical(
                "solver",
                ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
            ),
            "n_jobs": -1,
        }
        clf = LogisticRegression(**param)
        f1 = cross_val_score(
            clf, x_train, y_train, scoring="f1_weighted", cv=3, n_jobs=-1, verbose=0
        )
        return np.mean(f1)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    return study.best_params


if __name__ == "__main__":
    n_trials = 20

    knc_best_params = optimize_knc(n_trials=n_trials)
    rf_best_params = optimize_rf(n_trials=n_trials)
    lr_best_params = optimize_lr(n_trials=n_trials)

    print("\n", "=" * 30)
    print(f"knc best params : {knc_best_params}")
    print(f"rf best params : {rf_best_params}")
    print(f"lr best params : {lr_best_params}")
