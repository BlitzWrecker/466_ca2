# -*- coding: utf-8 -*-
import numpy as np
import struct
import matplotlib.pyplot as plt


def readMNISTdata():
    with open('t10k-images.idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows*ncols))

    with open('t10k-labels.idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size, 1))

    with open('train-images.idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows*ncols))

    with open('train-labels.idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size, 1))

    # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate(
        (np.ones([train_data.shape[0], 1]), train_data), axis=1)
    test_data = np.concatenate(
        (np.ones([test_data.shape[0], 1]),  test_data), axis=1)
    _random_indices = np.arange(len(train_data))
    np.random.shuffle(_random_indices)
    train_labels = train_labels[_random_indices]
    train_data = train_data[_random_indices]

    X_train = train_data[:50000] / 256
    t_train = train_labels[:50000]

    X_val = train_data[50000:] / 256
    t_val = train_labels[50000:]

    return X_train, t_train, X_val, t_val, test_data / 256, test_labels


def predict(X, W, t=None):
    # X_new: Nsample x (d+1)
    # W: (d+1) x K

    # TODO Your code here
    z = np.matmul(X, W)
    z_max = np.reshape(np.max(z, axis=1), (-1, 1))
    z_tilde = z - z_max
    y_tilde = np.exp(z_tilde)
    Z = np.reshape(np.sum(y_tilde, axis=1), (-1, 1))
    y = np.divide(y_tilde, Z)
    prediction = np.reshape(np.argmax(y, axis=1), (-1, 1))
    t_hat = np.zeros((t.shape[0], W.shape[1]))
    for i in range(len(t)):
        label = t[i][0]
        t_hat[i][label] = 1

    tmp = np.sum(np.multiply(t_hat, np.log(y)), axis=1)
    loss = -np.mean(tmp)
    acc = np.mean((prediction == t).astype(int))

    return y, t_hat, loss, acc


def train(X_train, y_train, X_val, t_val, adagrad=False):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    # TODO Your code here
    # initialization
    w = np.zeros([X_train.shape[1], N_class])
    # w: (d+1) x K

    # AdaGrad
    G = np.zeros([X_train.shape[1], N_class])
    # AdaGrad learning rates: (d+1) x K
    epsilon = 1e-10

    train_losses = []
    valid_accs = []

    W_best = None
    acc_best = 0
    epoch_best = 0

    for epoch in range(MaxEpoch):
        loss_this_epoch = 0

        for b in range(int(np.ceil(N_train / batch_size))):
            X_batch = X_train[b * batch_size: (b + 1) * batch_size]
            t_batch = y_train[b * batch_size: (b + 1) * batch_size]

            y_batch, t_batch_one_hot, loss_batch, _ = predict(X_batch, w, t_batch)
            loss_this_epoch += loss_batch

            gradient = np.matmul(np.transpose(X_batch), y_batch - t_batch_one_hot) / X_batch.shape[0]
            lr = alpha  # learning rate

            if adagrad:
                G += gradient * gradient
                lr = alpha / (np.sqrt(G) + epsilon)

            # Mini-batch gradient descent
            w = w - lr * gradient

        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        train_losses.append(loss_this_epoch)

        # 2. Perform validation on the validation set by the risk
        _, _, _, acc = predict(X_val, w, t_val)
        valid_accs.append(acc)

        # 3. Keep track of the best validation epoch, risk, and the weights
        if acc > acc_best:
            acc_best = acc
            epoch_best = epoch
            W_best = w

    return epoch_best, acc_best,  W_best, train_losses, valid_accs


##############################
# Main code starts here
X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata()


print(X_train.shape, t_train.shape, X_val.shape,
      t_val.shape, X_test.shape, t_test.shape)


N_class = 10

alpha = 0.1      # learning rate
batch_size = 100    # batch size
MaxEpoch = 50        # Maximum epoch
decay = 0.          # weight decay


# TODO: report 3 number, plot 2 curves
epoch_best, acc_best, W_best, train_losses, valid_accs = train(X_train, t_train, X_val, t_val)

_, _, _, acc_test = predict(X_test, W_best, t_test)

ag_epoch_best, ag_acc_best, ag_W_best, ag_train_losses, ag_valid_accs = (
    train(X_train, t_train, X_val, t_val, adagrad=True))

_, _, _, ag_acc_test = predict(X_test, ag_W_best, t_test)

# Report numbers and draw plots as required.
print('===== Part 1 (Baseline) =====')
print("The number of epoch that yielded the best validation performance: %s" % epoch_best)
print("The validation performance (accuracy) in that epoch: %f" % acc_best)
print("The test performance (accuracy) in that epoch: %f" % acc_test)

print('\n===== Part 2 (AdaGrad) =====')
print("The number of epoch that yielded the best validation performance: %s" % ag_epoch_best)
print("The validation performance (accuracy) in that epoch: %f" % ag_acc_best)
print("The test performance (accuracy) in that epoch: %f" % ag_acc_test)

plt.figure()
plt.title("Learning Curve of the Training Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Training Loss")
plt.plot(train_losses, label='Baseline')
plt.plot(ag_train_losses, label='AdaGrad')
plt.legend()

plt.figure()
plt.title("Learning Curve of the Validation Accuracy")
plt.xlabel("Number of Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(valid_accs, label='Baseline')
plt.plot(ag_valid_accs, label='AdaGrad')
plt.legend()
plt.show()
