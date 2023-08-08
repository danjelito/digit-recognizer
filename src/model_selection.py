import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from pathlib import Path
import tqdm

# list all models to be tried
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict

import model
import config
import utils


def main():
    # load dataset
    x_train, y_train = utils.load_dataset(load_test_set= False)

    # augment dataset
    num_augmented = 1000
    x_train, y_train = utils.augment_image(x_train, y_train, num_augmented)

    # preprocessing
    model.pipeline.fit(x_train, y_train)
    x_train = model.pipeline.transform(x_train)

    # try different classifier
    classifiers = {
        "logreg": LogisticRegression(max_iter=1000),
        "nb": GaussianNB(),
        "svc": SVC(random_state=config.RANDOM_STATE),
        "rf": RandomForestClassifier(random_state=config.RANDOM_STATE),
        "knn": KNeighborsClassifier(),
        "sgd": SGDClassifier(random_state=config.RANDOM_STATE),
        "dt": DecisionTreeClassifier(random_state=config.RANDOM_STATE),
        "gb": GradientBoostingClassifier(random_state=config.RANDOM_STATE),
    }

    # cross val the models, append result to result df
    df_result = pd.DataFrame(
        columns=["model", "accuracy", "f1", "precision", "recall", "time_taken"]
    )

    for key, classifier in tqdm(classifiers.items()):
        model_name = classifier.__class__.__name__

        start_time = time.time()
        y_pred = cross_val_predict(classifier, x_train, y_train, cv=3)
        elapsed_time = (time.time() - start_time) / 3  # divide by 3 for 3 cv

        acc, f1, prec, rec = utils.get_score(y_train, y_pred)
        df_result = df_result.append(
            {
                "model": model_name,
                "accuracy": acc,
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "time_taken": elapsed_time,
            },
            ignore_index=True,
        )

    # save result
    df_result.to_csv(config.PATH_MODEL_SELECTION_RES, index=False)
    print(df_result)


# if not already run
if not Path(config.PATH_MODEL_SELECTION_RES).exists():
    main()
