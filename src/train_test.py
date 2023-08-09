from pathlib import Path
import pickle

from sklearn.model_selection import cross_val_predict

import model
import config
import utils


def main():

    # load dataset
    x_train, y_train, x_test, y_test = utils.load_dataset(load_test_set= True)

    # load model and pipeline
    knc= model.knc
    pipeline = model.pipeline

    # preprocessing
    pipeline.fit(x_train, y_train)
    x_train = pipeline.transform(x_train)
    x_test = pipeline.transform(x_test)

    # train and validate
    print('train')
    y_pred = cross_val_predict(knc, x_train, y_train, cv=3)
    utils.get_score(y_pred=y_pred, y_true= y_train, print_result=True)  

    # train on whole test set and test
    print('\ntest')
    knc.fit(x_train, y_train)
    y_pred = knc.predict(x_test)
    utils.get_score(y_pred=y_pred, y_true= y_test, print_result=True)  

    # save model and pipeline
    if not Path(config.PATH_MODEL).exists():
        
        with open(config.PATH_MODEL, "wb") as file: # model
            pickle.dump(knc, file)
        
        with open(config.PATH_PIPELINE, "wb") as file: # pipeline
            pickle.dump(pipeline, file)
    
        print('model saved')
    
    else:
        print('model exists')


if __name__ == "__main__":
    main()