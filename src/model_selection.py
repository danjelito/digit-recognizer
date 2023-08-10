import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from pathlib import Path

# list all models to be tried
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict, StratifiedKFold

import model
import config
import utils


def main(augment=False):
    # load dataset
    x_train, y_train = utils.load_dataset(load_test_set=False)

    if augment:
        # augment dataset
        num_augmented = 1000
        x_train, y_train = utils.augment_image(x_train, y_train, num_augmented)

    # preprocessing
    # TODO: uncomment after adding deep learning
    # model.pipeline.fit(x_train, y_train)
    # x_train = model.pipeline.transform(x_train)

    # try different classifier
    # TODO: uncomment after adding deep learning
    classifiers = {
        # "logreg": LogisticRegression(max_iter=1000),
        # "nb": GaussianNB(),
        # "rf": RandomForestClassifier(random_state=config.RANDOM_STATE),
        # "knn": KNeighborsClassifier(),
        # "dt": DecisionTreeClassifier(random_state=config.RANDOM_STATE),
        # TODO: add deep learning, lgb and xgb
    }

    # cross val the models, append result to result df
    df_result = pd.DataFrame(
        columns=[
            "fold",
            "model",
            "accuracy",
            "f1",
            "precision",
            "recall",
            "prediction_time",
        ]
    )

    for key, classifier in classifiers.items():
        model_name = classifier.__class__.__name__

        # run kfold
        kfold = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=config.RANDOM_STATE
        )
        for fold, (train_idx, val_idx) in enumerate(kfold.split(x_train, y_train)):
            # set train val set
            x_t, y_t = x_train[train_idx], y_train[train_idx]
            x_v, y_v = x_train[val_idx], y_train[val_idx]

            # fit
            classifier.fit(x_t, y_t)

            # predict
            start_time = time.time()
            y_pred = classifier.predict(x_v)
            elapsed_time = time.time() - start_time

            # get score and append
            acc, f1, prec, rec = utils.get_score(y_v, y_pred)
            df_result = pd.concat(
                [
                    df_result,
                    pd.DataFrame(
                        {
                            "fold": fold,
                            "model": model_name,
                            "accuracy": acc,
                            "f1": f1,
                            "precision": prec,
                            "recall": rec,
                            "prediction_time": elapsed_time,
                        },
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )

    # save result
    df_result = (
        df_result.groupby("model")
        .agg(
            {
                "accuracy": "mean",
                "f1": "mean",
                "precision": "mean",
                "recall": "mean",
                "prediction_time": "sum",
            }
        )
        .sort_values("f1", ascending=False)
    )
    # TODO: uncomment after adding deep learning
    # df_result.to_csv(config.PATH_MODEL_SELECTION_RES, index=False)
    print(df_result)


if __name__ == "__main__":
    # if not already run
    if not Path(config.PATH_MODEL_SELECTION_RES).exists():
        main(augment=False)
    else:
        df_result = pd.read_csv(config.PATH_MODEL_SELECTION_RES)
        df_result = df_result.sort_values("f1", ascending=False)
        print(df_result)
