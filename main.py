import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import time
from itertools import combinations

MAX_PASS = 500  # default
ONE_VS_ALL = 0
ONE_VS_ONE = 1
DIRECT = 2


def ReadX(path):
    print(f'>>> Reading data from: {path} ...')
    with open(path) as f:
        file = f.readlines()
    print(f'#instances: {len(file)}')

    X_all = []
    for instance in file:
        f = filter(None, instance.split(' '))
        instance_filtered = list(f)
        instance_cleaned = [float(attr.strip()) for attr in instance_filtered]
        X_all.append(instance_cleaned)
    X_all = np.array(X_all)

    return X_all


def ReadY(path):
    print(f'>>> Reading data from: {path} ...')
    with open(path) as f:
        file = f.readlines()
        print(f'#instances: {len(file)}')

    y_all = [float(label.strip()) for label in file]
    y_all = np.array(y_all)
    return y_all


def unison_shuffled_copies(a, b):
    """
    taken from: https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def binary_perceptron(X: np.array, y: np.array, w_init: np.array, b_init: int, max_pass: int) \
        -> (np.array, int, np.array):
    """
    :param X: training set of size (instances) n and dimension (attributes) d
    :param y: n training labels (-1 or 1) corresponding to the n vectors in X
    :param w_init: initialization of weight vector
    :param b_init: initialization of bias
    :param max_pass: max number of passes of the training set
    :return: weights, bias, mistakes (where mistake[t] = num mistakes during pass t)
    :param multi: true if perceptron is being used as multiclass
    """
    n, d = X.shape
    w, b, mistakes = w_init, b_init, np.zeros(max_pass)
    printProgressBar(0, max_pass, prefix='Progress:', suffix='Complete', length=50)

    for t in range(max_pass):
        X, y = unison_shuffled_copies(X, y)  # randomize order of instances
        for X_i, y_i in zip(X, y):
            if y_i * (np.dot(X_i, w) + b) <= 0:  # mistake happened
                w += y_i * np.array(X_i).reshape(d,)
                b += y_i
                mistakes[t] += 1
        printProgressBar(t + 1, max_pass, prefix='Progress:', suffix='Complete', length=50)

    return w, b, mistakes


def predict(x: np.array, w: np.array, b: int) -> int:
    """
    Makes prediction given vector x and a perceptron
    :param x: data vector
    :param w: weight vector
    :param b: bias
    :return: y-hat (predicted value)
    """
    return np.dot(x, w) + b


def run_binary(max_pass, X: np.array, y: np.array, plot=True):
    """
    runs binary perceptron
    :param plot: if True, shows plot of errors w.r.t iteration
    """
    start = time.time()
    n, d = X.shape
    w = np.zeros(d)

    res_w, res_b, res_mistakes = binary_perceptron(X, y, w, 0, max_pass)
    error_rate = 100*res_mistakes[-1]/n

    print(f"Error rate after {max_pass} iterations is {error_rate:.2f}% "
          f"({res_mistakes[-1]} mistakes out of {n})\n")
    print(f"Time elapsed: {(time.time() - start):.2f} seconds")

    if plot:
        plt.plot(res_mistakes)
        plt.xlabel("Pass number")
        plt.ylabel("Mistakes")
        plt.show()


def isolate_class(y_train: np.array, target_class: int):
    """
    Auxiliary func for one-vs-all: Modifies label vector from multiclass to binary
    """
    mod_y = y_train.copy()
    for i in range(len(mod_y)):
        mod_y[i] = 1 if mod_y[i] == target_class else -1
    return mod_y


def isolate_classes(X_train: np.array, y_train: np.array, first_class: int, second_class: int):
    """
    Auxiliary func for one-vs-one: Modifies both data set and labels to include only the 2 target classes,
     and converts labels to +-1
    """
    mod_X = []
    mod_y = []
    for X_i, y_i in zip(X_train, y_train):
        if y_i in (first_class, second_class):
            mod_y.append(1 if y_i == first_class else -1)
            mod_X.append(X_i)
    return np.array(mod_X), np.array(mod_y)


def best_prediction(x: np.array, c: int, weights: np.array, biases: np.array, multi_type: int):
    """
    For multiclass: given a set of perceptrons and an input x, returns the "best" class prediction
    according to maximization of y-hat
    :param x: data vector
    :param c: num classes
    :param weights: either c (if one-vs-all) or c-choose-2 (if one-vs-one) weight vectors
    :param biases: c or c-choose-2 biases
    :param multi_type: ONE_VS_ALL or ONE_VS_ONE
    :return: the best prediction (out of classes 1 to c)
    """
    if multi_type is ONE_VS_ALL:
        predictions = []
        for w_k, b_k in zip(weights, biases):
            predictions.append(predict(x, w_k, b_k))
        return np.argmax(predictions) + 1

    if multi_type is ONE_VS_ONE:
        pairs = list(combinations(list(range(1, c + 1)), 2))
        histogram = np.zeros(c+1)  # histogram
        for i, pair in enumerate(pairs):
            prediction = np.sign(predict(x, weights[i], biases[i]))
            if prediction >= 0:
                histogram[pair[0]] += 1
            else:
                histogram[pair[1]] += 1
        return np.argmax(histogram)

    if multi_type is DIRECT:
        return predict_multi(np.append(x, 1), np.c_[weights, biases], c)


def get_error(y_true: np.array, y_predicted: np):
    """
    Returns number of mistakes and percentage
    """
    assert len(y_true) == len(y_predicted)
    mistakes = 0
    for i, label in enumerate(y_true):
        if label != y_predicted[i]:
            mistakes += 1
    return mistakes, mistakes*100/len(y_true)


def run_multiclass(c, multi_type, max_pass, X_train, y_train, X_test, y_test):
    """
    runs multiclass perceptron on activity dataset
    :param multi_type: ONE_VS_ALL or ONE_VS_ONE
    :param c: number of classes
    """
    start = time.time()

    # training
    n, d = X_train.shape  # training set has n instances with d attributes each
    weights = []  # will contain c arrays of weights
    biases = []  # will contain c b-values

    if multi_type is ONE_VS_ALL:
        for k in range(c):
            print(f"Training perceptron {k+1}-vs-rest")
            y_train_binary = isolate_class(y_train, k+1)
            w, b, mistakes = binary_perceptron(X_train, y_train_binary, np.zeros(d), 0, max_pass)
            weights.append(w)
            biases.append(b)

    if multi_type is ONE_VS_ONE:
        classes = list(range(1, c + 1))
        pairs = list(combinations(classes, 2))
        for k in pairs:
            print(f"Training perceptron {k[0]}-vs-{k[1]}")
            X_train_binary, y_train_binary = isolate_classes(X_train, y_train, k[0], k[1])
            w, b, mistakes = binary_perceptron(X_train_binary, y_train_binary, np.zeros(d), 0, max_pass)
            weights.append(w)
            biases.append(b)

    if multi_type is DIRECT:
        weights, biases, mistakes = direct_multiclass(X_train, y_train, c, max_pass)

    # predicting on training data
    y_train_predicted = []
    for x_i in X_train:
        y_train_predicted.append(best_prediction(x_i, c, weights, biases, multi_type))

    # predicting on testing data
    y_test_predicted = []
    for x_i in X_test:
        y_test_predicted.append(best_prediction(x_i, c, weights, biases, multi_type))

    train_errors, train_error_rate = get_error(y_train, y_train_predicted)
    test_errors, test_error_rate = get_error(y_test, y_test_predicted)

    print(f"For max_pass={max_pass}, train set error rate is {train_error_rate:.2f}% "
          f"({train_errors} mistakes out of {n})\n"
          f"and test set error rate is {test_error_rate:.2f}% "
          f"({test_errors} mistakes out of {len(y_test)})\n")
    print(f"Time elapsed: {(time.time() - start):.2f} seconds")


def direct_multiclass(X: np.array, y: np.array, c: int, max_pass: int):
    """
    For exercise 1.6 - direct multiclass perceptron
    """
    n, d = X.shape  # training set has n instances with d attributes each
    X = np.c_[X, np.ones(n)]  # pad each x_i with 1
    W = np.zeros((c, d+1))
    mistakes = np.zeros(max_pass)

    printProgressBar(0, max_pass, prefix='Progress:', suffix='Complete', length=50)

    for t in range(max_pass):
        X, y = unison_shuffled_copies(X, y)  # randomize order of instances
        for x_i, y_i in zip(X, y):
            y_hat = predict_multi(x_i, W, c)
            if y_hat is not y_i:
                W[y_hat - 1] -= x_i
                W[int(y_i) - 1] += x_i
                mistakes[t] += 1
        printProgressBar(t + 1, max_pass, prefix='Progress:', suffix='Complete', length=50)

    weights, biases = W[:, :-1], W[:, -1]
    return weights, biases, mistakes


def predict_multi(x_i: np.array, W: np.array, c: int):
    """
    returns class k whose corresponding w_k maximizes the inner product with x_i
    """
    dot_products = np.zeros(c)
    for k, w_k in enumerate(W):
        dot_products[k] = np.dot(x_i, w_k)
    return np.argmax(dot_products) + 1


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Taken from https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == '__main__':

    if len(sys.argv) not in (2, 3):
        print('Error: invalid argument')
        exit(1)

    arg = sys.argv[1]
    max_pass = int(sys.argv[2]) if len(sys.argv) == 3 else MAX_PASS

    if arg == '1':
        data = np.array(pd.read_csv('spambase_data/spambase_X.csv', header=None).to_numpy()).transpose()
        labels = pd.read_csv('spambase_data/spambase_y.csv', header=None).to_numpy()
        run_binary(max_pass, data, labels)

    else:
        num_classes = 6
        train_data = ReadX('activity_data/activity_X_train.txt')
        train_labels = ReadY('activity_data/activity_y_train.txt')
        test_data = ReadX('activity_data/activity_X_test.txt')
        test_labels = ReadY('activity_data/activity_y_test.txt')

        if arg == '2':
            run_multiclass(num_classes, ONE_VS_ALL, max_pass, train_data, train_labels, test_data, test_labels)
        elif arg == '3':
            run_multiclass(num_classes, ONE_VS_ONE, max_pass, train_data, train_labels, test_data, test_labels)
        elif arg == '4':
            run_multiclass(num_classes, DIRECT, max_pass, train_data, train_labels, test_data, test_labels)

        else:
            print('Error: invalid argument')
            exit(1)


