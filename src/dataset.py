from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
import config


if not Path(config.PATH_X_TRAIN).exists():

    mnist = fetch_openml("mnist_784", as_frame=False, parser="auto", version=1)

    x = mnist["data"]
    y = mnist["target"]

    # shuffle dataset
    num_samples = y.shape[0]
    permutation = np.random.permutation(num_samples)
    x = x[permutation]
    y = y[permutation]

    # create train test split
    test_size = 0.1
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify= y, 
        random_state=config.RANDOM_STATE
    )

    # save dataset
    np.save(file= config.PATH_X_TRAIN, arr= x_train, allow_pickle= True)
    np.save(file= config.PATH_Y_TRAIN, arr= y_train, allow_pickle= True)
    np.save(file= config.PATH_X_TEST, arr= x_test, allow_pickle= True)
    np.save(file= config.PATH_Y_TEST, arr= y_test, allow_pickle= True)
    
    print('file saved')

else:
    print('file exists')
