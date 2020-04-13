import numpy as np

Y = np.random.randint(1, 10, (10, 1))


def one_hot(Y):
    Y = Y - Y.min()
    return (Y == np.arange(Y.max() + 1)).astype(int)

print(one_hot(Y))