import h5py
import matplotlib.pyplot as plt
import numpy as np


def opencurve():

    file = h5py.File('./utils/info_l2_50.h5', 'r')
    train_loss = np.array(file['train_loss'])
    val_loss = np.array(file['val_loss'])

    train_acc = np.array(file['train_acc'])
    val_acc = np.array(file['val_acc'])

    plt.figure(1)
    plt.plot(range(50), train_loss, label='train_loss')
    plt.plot(range(50), val_loss, label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc='upper right')


    plt.figure(2)
    plt.plot(range(50), train_acc, label='train_acc')
    plt.plot(range(50), val_acc, label='val_acc')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':

    opencurve()