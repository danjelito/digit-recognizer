import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, cross_validate
import optuna

import model
import config
import utils


def main():

    # load dataset
    x_train, y_train, x_test, y_test = utils.load_dataset(load_test_set= True)

    # load model
    knc= model.knc

    # preprocessing
    model.pipeline.fit(x_train, y_train)
    x_train = model.pipeline.transform(x_train)
    x_test = model.pipeline.transform(x_test)

    # train and validate
    print('train')
    y_pred = cross_val_predict(knc, x_train, y_train, cv=3)
    utils.get_score(y_pred=y_pred, y_true= y_train, print_result=True)  

    # train on whole test set and test
    print('\ntest')
    knc.fit(x_train, y_train)
    y_pred = knc.predict(x_test)
    utils.get_score(y_pred=y_pred, y_true= y_test, print_result=True)  

if __name__ == "__main__":
    main()