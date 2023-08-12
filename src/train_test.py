from pathlib import Path
import pickle

from sklearn.model_selection import cross_val_predict

import model
import config
import utils


def save_sklearn_object(object, path):
    if Path(path).exists():
        print(f"object already exists at {path}")
    else:
        with open(path, "wb") as file:
            pickle.dump(object, file)
        print(f"object saved at {path}")


def load_dataset():
    # load dataset
    x_train, y_train, x_test, y_test = utils.load_dataset(load_test_set=True)
    return x_train, y_train, x_test, y_test


def preprocess_dataset(x_train, y_train, x_test, y_test, save_path=None):
    # preprocessing
    pipeline = model.pipeline
    pipeline.fit(x_train, y_train)
    x_train = pipeline.transform(x_train)
    x_test = pipeline.transform(x_test)

    if save_path is not None:
        save_sklearn_object(pipeline, config.PATH_PIPELINE)

    return x_train, y_train, x_test, y_test


def train_and_test(model, x_train, y_train, x_test, y_test, save_path=None):
    model_name = model_name = model.__class__.__name__
    print("=" * 15 , model_name, "=" * 15)

    print("train")
    # cross validate on test set
    y_pred = cross_val_predict(model, x_train, y_train, cv=3)
    utils.get_score(y_pred=y_pred, y_true=y_train, print_result=True)

    print("test")
    # fit on train and predict on test
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    utils.get_score(y_pred=y_pred, y_true=y_test, print_result=True)

    # save model
    if save_path is not None:
        save_sklearn_object(model, save_path)


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_dataset()
    x_train, y_train, x_test, y_test = preprocess_dataset(
        x_train, y_train, x_test, y_test, save_path=config.PATH_PIPELINE
    )

    # train and test each model
    # save model to each path after training
    train_and_test(
        model.knc, x_train, y_train, x_test, y_test, save_path=config.PATH_MODEL_KNC
    )
    train_and_test(
        model.rf, x_train, y_train, x_test, y_test, save_path=config.PATH_MODEL_RF
    )
    train_and_test(
        model.lr, x_train, y_train, x_test, y_test, save_path=config.PATH_MODEL_LR
    )
