import numpy as np


def step_gradient_descent(W, b, dW, db, alpha):
    """Single iteration of gradient descent.

    Args:
        W (ndarray[n, 1]):      weights for current step
        b (ndarray[1, 1]):      bias for current step
        dW (ndarray[n, 1]):     weight gradients for current step
        db (ndarray[1, 1]):     bias gradient for current step
        alpha (float):          learning rate

    Returns:
        W (ndarray[n, 1]):      updated weights
        b (ndarray[1, 1]):      updated bias
    """
    return W - alpha * dW, b - alpha * db


def step_momentum(W, b, dW, db, alpha, v, beta=0.9):
    """Single iteration of gradient descent with momentum.
    NB! In-place modification of 'v'.

    Args:
        W (ndarray[n, 1]):          weights for current step
        b (ndarray[1, 1]):          bias for current step
        dW (ndarray[n, 1]):         weight gradients for current step
        db (ndarray[1, 1]):         bias gradient for current step
        alpha (float):              learning rate
        v (ndarray[1, n]):          moment (exp. weighted average)
        beta (float, optional):     moment decay rate (defaults to 0.9)

    Returns:
        W (ndarray[n, 1]):          updated weights
        b (ndarray[1, 1]):          updated bias
    """
    v[0][...] = beta * v[0] + (1 - beta) * dW
    v[1][...] = beta * v[1] + (1 - beta) * db
    return W - alpha * v[0], b - alpha * v[1]


def step_rmsprop(W, b, dW, db, alpha, s, beta=0.9, epsilon=10e-8):
    """Single iteration of gradient descent using RMSprop.
    NB! In-place modification of 's'.

    Args:
        W (ndarray[n, 1]):          weights for current step
        b (ndarray[1, 1]):          bias for current step
        dW (ndarray[n, 1]):         weight gradients for current step
        db (ndarray[1, 1]):         bias gradient for current step
        alpha (float):              learning rate
        s (ndarray[1, n]):          moment (exp. weighted average)
        beta (float, optional):     moment decay rate (defaults to 0.9)
        epsilon (float, optional):  numerical stability constant (defaults to 10e-8)

    Returns:
        W (ndarray[n, 1]):          updated weights
        b (ndarray[1, 1]):          updated bias
    """
    s[0][...] = beta * s[0] + (1 - beta) * np.square(dW)
    s[1][...] = beta * s[1] + (1 - beta) * np.square(db)
    W = W - alpha * dW/(np.sqrt(s[0]) + epsilon)
    b = b - alpha * db/(np.sqrt(s[1]) + epsilon)
    return W, b


def step_adam(W, b, dW, db, alpha, v, s, t, beta1=0.9, beta2=0.999, epsilon=10e-8):
    """Single iteration of gradient descent using Adam.
    NB! In-place modification of 'v', 's' and 't'.

    Args:
        W (ndarray[n, 1]):          weights for current step
        b (ndarray[1, 1]):          bias for current step
        dW (ndarray[n, 1]):         weight gradients for current step
        db (ndarray[1, 1]):         bias gradient for current step
        alpha (float):              learning rate
        v (ndarray[1, n]):          1st order moment (exp. weighted average)
        s (ndarray[1, n]):          2nd order moment (exp. weighted average)
        t (int):                    iteration count for bias correction
        beta1 (float, optional):    1st order moment decay rate (defaults to 0.9)
        beta1 (float, optional):    2nd order moment decay rate (defaults to 0.999)
        epsilon (float, optional):  numerical stability constant (defaults to 10e-8)

    Returns:
        W (ndarray[n, 1]):          updated weights
        b (ndarray[1, 1]):          updated bias
    """
    t[...] = t + 1
    v[0][...] = beta1 * v[0] + (1 - beta1) * dW
    v[1][...] = beta1 * v[1] + (1 - beta1) * db
    s[0][...] = beta2 * s[0] + (1 - beta2) * np.square(dW)
    s[1][...] = beta2 * s[1] + (1 - beta2) * np.square(db)
    v0_corr = v[0] / (1 - np.power(beta1, t))
    v1_corr = v[1] / (1 - np.power(beta1, t))
    s0_corr = s[0] / (1 - np.power(beta2, t))
    s1_corr = s[1] / (1 - np.power(beta2, t))
    W = W - alpha * v0_corr/(np.sqrt(s0_corr) + epsilon)
    b = b - alpha * v1_corr/(np.sqrt(s1_corr) + epsilon)
    return W, b
