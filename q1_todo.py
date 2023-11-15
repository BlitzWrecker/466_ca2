from utils import plot_data, generate_data
import numpy as np


"""
Documentation:

Function generate() takes as input "A" or "B", it returns X, t.
X is two dimensional vectors, t is the list of labels (0 or 1).    

Function plot_data(X, t, w=None, bias=None, is_logistic=False, figure_name=None)
takes as input pairs of (X, t) , parameter w, and bias. 
If you are plotting the decision boundary for a logistic classifier, set "is_logistic" as True
"figure_name" specifies the name of the saved diagram.
"""


def train_logistic_regression(X, t):
    """
    Given data, train your logistic classifier.
    Return weight and bias
    """
    t = np.reshape(t, (-1, 1))
    M = X.shape[0]
    alpha = 0.1
    num_of_epochs = 1000
    batch_size = 10

    w = np.zeros((X.shape[1], 1))
    b = 0

    np.random.seed(314)
    np.random.shuffle(X)
    np.random.seed(314)
    np.random.shuffle(t)

    for epoch in range(num_of_epochs):
        for b in range(int(np.ceil(M / batch_size))):
            X_batch = X[b * batch_size: (b + 1) * batch_size]
            t_batch = t[b * batch_size: (b + 1) * batch_size]

            z = np.matmul(X_batch, w) + b
            y = 1 / (1 + np.exp(-1 * z))

            w = w - alpha * np.matmul(np.transpose(X_batch), y - t_batch)
            b = b - alpha * np.sum(y - t_batch)

    return w, b


def predict_logistic_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """
    t = np.matmul(X, w) + b
    t[t >= 0] = 1
    t[t < 0] = 0
    t = np.reshape(t, (-1,))

    return t


def train_linear_regression(X, t):
    """
    Given data, train your linear regression classifier.
    Return weight and bias
    """
    X_ = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    X_T = np.transpose(X_)

    w_tilde = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_T, X_)), X_T), t)
    w = w_tilde[1:3]
    b = w_tilde[0]

    return w, b


def predict_linear_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """
    t = np.matmul(X, w) + b
    t[t < 0.5] = 0
    t[t >= 0.5] = 1

    return t


def get_accuracy(t, t_hat):
    """
    Calculate accuracy,
    """
    M = t.shape[0]
    assert M == t_hat.shape[0]
    acc = np.sum((t == t_hat).astype(int)) / M
    return acc


def main():
    # Dataset A
    # Linear regression classifier
    X, t = generate_data("A")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False,
              figure_name='dataset_A_linear.png')

    # logistic regression classifier
    X, t = generate_data("A")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of logistic regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True,
              figure_name='dataset_A_logistic.png')

    # Dataset B
    # Linear regression classifier
    X, t = generate_data("B")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False,
              figure_name='dataset_B_linear.png')

    # logistic regression classifier
    X, t = generate_data("B")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of logistic regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True,
              figure_name='dataset_B_logistic.png')


main()
