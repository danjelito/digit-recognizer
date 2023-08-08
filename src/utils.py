import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import shift, rotate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import config


def load_dataset():
    x_train = np.load(config.PATH_X_TRAIN, allow_pickle=True)
    y_train = np.load(config.PATH_Y_TRAIN, allow_pickle=True)
    x_test = np.load(config.PATH_X_TEST, allow_pickle=True)
    y_test = np.load(config.PATH_Y_TEST, allow_pickle=True)
    return x_train, y_train, x_test, y_test


def plot_digits(arr, label_arr):
    num_samples = arr.shape[0]
    num_rows = 10
    num_cols = 10

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    axes = axes.ravel()

    for i in range(num_rows * num_cols):
        img = arr[i].reshape((28, 28))
        label = label_arr[i]

        axes[i].imshow(img, cmap="gray_r")
        axes[i].axis("off")
        axes[i].set_title(label, fontsize=8, color="red")

    plt.subplots_adjust(wspace=0.0, hspace=1.0)
    plt.show()


def shift_image(image, dx, dy):
    """Shift image"""
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])


def rotate_image(image, angle):
    """Rotate image"""
    image = image.reshape((28, 28))
    rotated_image = rotate(image, angle, reshape= False)
    return rotated_image.reshape([-1])


def augment_image(x_train, y_train, num_augmented):
    
    # shuffle
    permutation = np.random.permutation(x_train.shape[0])
    x_train = x_train[permutation]
    y_train = y_train[permutation]

    # shift
    x_train_shifted = []
    y_train_shifted = []
    for dx, dy in ((3, 0), (-3, 0), (0, 3), (0, -3)):
        for x, y in zip(x_train[:num_augmented], y_train[:num_augmented]):
            x_shifted = shift_image(image= x, dx= dx, dy= dy)
            x_train_shifted.append(x_shifted)
            y_train_shifted.append(y)
    x_train_shifted = np.array(x_train_shifted)
    y_train_shifted = np.array(y_train_shifted)

    # rotate
    x_train_rotated = []
    y_train_rotated = []
    for angle in [10, 20, 30]:
        for x, y in zip(x_train[-num_augmented:], y_train[-num_augmented:]):
            x_rot = rotate_image(image= x, angle= angle)

            # rescale the pixel values to be in the range [0, 255]
            num = (x_rot - np.min(x_rot))
            den = (np.max(x_rot) - np.min(x_rot))
            x_rot = num // den * 255
            
            x_train_rotated.append(x_rot)
            y_train_rotated.append(y)
    x_train_rotated = np.array(x_train_rotated)
    y_train_rotated = np.array(y_train_rotated)

    # concat all arrays together
    x_train = np.concatenate((x_train, x_train_shifted, x_train_rotated), axis=0)
    y_train = np.concatenate((y_train, y_train_shifted, y_train_rotated), axis=0)

    # shuffle again
    permutation = np.random.permutation(x_train.shape[0])
    x_train = x_train[permutation]
    y_train = y_train[permutation]

    return x_train, y_train



def get_score(y_true, y_pred, print_result= False):
    """Get and print accuracy, f1, percision and recall"""
    
    try:
        nunique= len(np.unique(y_true))
    except: 
        nunique= y_true.nunique()

    average = 'weighted' if nunique > 2 else 'binary'

    acc= accuracy_score(y_true, y_pred)
    f1= f1_score(y_true, y_pred, average= average)
    prec= precision_score(y_true, y_pred, average= average)
    rec= recall_score(y_true, y_pred, average= average)

    if print_result:

        names = ['accuracy', 'f1', 'precision', 'recall']
        values = [acc, f1, prec, rec]

        for i, (name, value) in enumerate(zip(names, values)):
            print(f"{i+1: 2}) {name: <30s} {value: .4f}")

    return acc, f1, prec, rec