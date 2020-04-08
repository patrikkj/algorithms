import numpy as np

from .helpers import (step_adam, step_gradient_descent, step_momentum,
                      step_rmsprop)


def mini_batch_gradient_descent(params, X, y, cost_func, grad_func, 
                                alpha=0.01, epochs=100, k=64, l=0, 
                                step_func=None, **hparams):
    """Minimizes the objective function using mini-batch gradient descent.

    Args:
        params (ndarray[1, n]):             initial parameters 
        X (ndarray[m, ...]):                input features
        y (ndarray[m, 1]):                  output labels
        cost_func (... -> float32):         mapping of the form (params, X, Y) -> cost
        grad_func (... -> ndarray[1, n]):   gradients of cost function
        alpha (float, optional):            learning rate (defaults to 0.01)
        epochs (int, optional):             number of iterations (defaults to 100)
        k (int, optional):                  mini-batch size (defaults to 64)
        l (float, optional):                regularization parameter (defaults to 0)

    Returns:
        params (ndarray[1, n]):             updated parameters
        costs (ndarray[epochs, 1]):         costs for each batch iteration
        grads (ndarray[n, epochs]):         gradients for each batch iteration
    """
    # Initialization
    if step_func is None:
        step_func = step_gradient_descent
    costs, grads = [], []

    for _ in range(epochs):
        # Shuffle input and labels
        m = X.shape[0]
        p = np.random.permutation(m)
        X, y = X[p], y[p]

        # Partition input and labels
        X_batches = np.split(X, range(k, m, k))
        y_batches = np.split(y, range(k, m, k))

        # Perform a single iteration of gradient descent for every mini-batch
        for X_batch, y_batch in zip(X_batches, y_batches):
            grad = grad_func(params, X_batch, y_batch, l)
            cost = cost_func(params, X_batch, y_batch, l)
            params = step_func(params, grad, alpha, **hparams)
            costs.append(cost)
            grads.append(grad)
    return params, np.array(costs), np.array(grads)


def batch_gradient_descent(params, X, y, cost_func, grad_func, 
                           alpha=0.01, epochs=100, l=0):
    """Minimizes the objective function using batch gradient descent.

    Args:
        params (ndarray[1, n]):             initial parameters 
        X (ndarray[m, ...]):                input features
        y (ndarray[m, 1]):                  output labels
        cost_func (... -> float32):         mapping of the form (params, X, Y) -> cost
        grad_func (... -> ndarray[1, n]):   gradients of cost function
        alpha (float, optional):            learning rate (defaults to 0.01)
        epochs (int, optional):             number of iterations (defaults to 100)
        l (float, optional):                regularization parameter (defaults to 0)

    Returns:
        params (ndarray[1, n]):             updated parameters
        costs (ndarray[epochs, 1]):         costs for each batch iteration
        grads (ndarray[n, epochs]):         gradients for each batch iteration
    """
    return mini_batch_gradient_descent(params, X, y, cost_func, grad_func, alpha, epochs, X.shape[0], l)


def stochastic_gradient_descent(params, X, y, cost_func, grad_func, alpha=0.01, epochs=100, l=0):
    """Minimizes the objective function using stochastic gradient descent.

    Args:
        params (ndarray[1, n]):             initial parameters 
        X (ndarray[m, ...]):                input features
        y (ndarray[m, 1]):                  output labels
        cost_func (... -> float32):         mapping of the form (params, X, Y) -> cost
        grad_func (... -> ndarray[1, n]):   gradients of cost function
        alpha (float, optional):            learning rate (defaults to 0.01)
        epochs (int, optional):             number of iterations (defaults to 100)
        l (float, optional):                regularization parameter (defaults to 0)

    Returns:
        params (ndarray[1, n]):             updated parameters
        costs (ndarray[epochs, 1]):         costs for each batch iteration
        grads (ndarray[n, epochs]):         gradients for each batch iteration
    """
    return mini_batch_gradient_descent(params, X, y, cost_func, grad_func, alpha, epochs, 1, l)


def momentum_gradient_descent(params, X, y, cost_func, grad_func, 
                              alpha=0.01, epochs=100, k=64, l=0, beta=0.9):
    """Minimizes the objective function using mini-batch gradient descent w/ momentum.

    Args:
        params (ndarray[1, n]):             initial parameters 
        X (ndarray[m, ...]):                input features
        y (ndarray[m, 1]):                  output labels
        cost_func (... -> float32):         mapping of the form (params, X, Y) -> cost
        grad_func (... -> ndarray[1, n]):   gradients of cost function
        alpha (float, optional):            learning rate (defaults to 0.01)
        epochs (int, optional):             number of iterations (defaults to 100)
        k (int, optional):                  mini-batch size (defaults to 64)
        l (float, optional):                regularization parameter (defaults to 0)
        beta (float, optional):             moment decay rate (defaults to 0.9)

    Returns:
        params (ndarray[1, n]):             updated parameters
        costs (ndarray[epochs, 1]):         costs for each batch iteration
        grads (ndarray[n, epochs]):         gradients for each batch iteration
    """
    args = (params, X, y, cost_func, grad_func)
    kwargs = {
        'alpha': alpha,
        'epochs': epochs,
        'k': k,
        'l': l,
        'step_func': step_momentum
    }
    hparams = {  # Hyperparameters passed to the step function
        'v': np.zeros(params.shape),
        'beta': beta
    }
    return mini_batch_gradient_descent(*args, **kwargs, **hparams)


def rmsprop(params, X, y, cost_func, grad_func, 
            alpha=0.01, epochs=100, k=64, l=0, beta=0.9):
    """Minimizes the objective function using RMSprop.

    Args:
        params (ndarray[1, n]):             initial parameters 
        X (ndarray[m, ...]):                input features
        y (ndarray[m, 1]):                  output labels
        cost_func (... -> float32):         mapping of the form (params, X, Y) -> cost
        grad_func (... -> ndarray[1, n]):   gradients of cost function
        alpha (float, optional):            learning rate (defaults to 0.01)
        epochs (int, optional):             number of iterations (defaults to 100)
        k (int, optional):                  mini-batch size (defaults to 64)
        l (float, optional):                regularization parameter (defaults to 0)
        beta (float, optional):             moment decay rate (defaults to 0.9)

    Returns:
        params (ndarray[1, n]):             updated parameters
        costs (ndarray[epochs, 1]):         costs for each batch iteration
        grads (ndarray[n, epochs]):         gradients for each batch iteration
    """
    args = (params, X, y, cost_func, grad_func)
    kwargs = {
        'alpha': alpha,
        'epochs': epochs,
        'k': k,
        'l': l,
        'step_func': step_rmsprop
    }
    hparams = {  # Hyperparameters passed to the step function
        's': np.zeros(params.shape),
        'beta': beta
    }
    return mini_batch_gradient_descent(*args, **kwargs, **hparams)


def adam(params, X, y, cost_func, grad_func, 
         alpha=0.01, epochs=100, k=64, l=0, 
         beta1=0.9, beta2=0.999, epsilon=10e-8):
    """Minimizes the objective function using Adam.

    Args:
        params (ndarray[1, n]):             initial parameters 
        X (ndarray[m, ...]):                input features
        y (ndarray[m, 1]):                  output labels
        cost_func (... -> float32):         mapping of the form (params, X, Y) -> cost
        grad_func (... -> ndarray[1, n]):   gradients of cost function
        alpha (float, optional):            learning rate (defaults to 0.01)
        epochs (int, optional):             number of iterations (defaults to 100)
        k (int, optional):                  mini-batch size (defaults to 64)
        l (float, optional):                regularization parameter (defaults to 0)
        beta1 (float, optional):            1st order moment decay rate (defaults to 0.9)
        beta2 (float, optional):            2nd order moment decay rate (defaults to 0.999)
        epsilon (float, optional):          numerical stability constant (defaults to 10e-8)

    Returns:
        params (ndarray[1, n]):             updated parameters
        costs (ndarray[epochs, 1]):         costs for each batch iteration
        grads (ndarray[n, epochs]):         gradients for each batch iteration
    """
    args = (params, X, y, cost_func, grad_func)
    kwargs = {
        'alpha': alpha,
        'epochs': epochs,
        'k': k,
        'l': l,
        'step_func': step_adam
    }
    hparams = {  # Hyperparameters passed to the step function
        'v': np.zeros(params.shape),
        's': np.zeros(params.shape),
        't': np.array([0]),
        'beta1': beta1,
        'beta2': beta2,
        'epsilon': epsilon
    }
    return mini_batch_gradient_descent(*args, **kwargs, **hparams)
