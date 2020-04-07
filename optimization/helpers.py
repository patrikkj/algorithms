import numpy as np


def step_gradient_descent(params, grad, alpha):
    """Single iteration of gradient descent.
    
    Args:
        params (ndarray[1, n]): parameters for current step
        grad (ndarray[1, n]):   gradient for current step
        alpha (float):          learning rate
    
    Returns:
        params (ndarray[1, n]): updated parameters
    """
    return params - alpha * grad

def step_momentum(params, grad, alpha, v, beta=0.9):
    """Single iteration of gradient descent with momentum.
    NB! In-place modification of 'v'.
    
    Args:
        params (ndarray[1, n]): parameters for current step
        grad (ndarray[1, n]):   gradient for current step
        alpha (float):          learning rate
        v (ndarray[1, n]):      moment (exp. weighted average)
        beta (float, optional): moment decay rate (defaults to 0.9)
    
    Returns:
        params (ndarray[1, n]): updated parameters
    """
    v[...] = beta * v + (1 - beta) * grad
    return params - alpha * v

def step_rmsprop(params, grad, alpha, s, beta=0.9, epsilon=10e-8):
    """Single iteration of gradient descent using RMSprop.
    NB! In-place modification of 's'.
    
    Args:
        params (ndarray[1, n]):     parameters for current step
        grad (ndarray[1, n]):       gradient for current step
        alpha (float):              learning rate
        s (ndarray[1, n]):          moment (exp. weighted average)
        beta (float, optional):     moment decay rate (defaults to 0.9)
        epsilon (float, optional):  numerical stability constant (defaults to 10e-8)
    
    Returns:
        params (ndarray[1, n]):     updated parameters
    """
    s[...] = beta * s + (1 - beta) * np.square(grad)
    return params - alpha * grad/(np.sqrt(s) + epsilon)

def step_adam(params, grad, alpha, v, s, t, beta1=0.9, beta2=0.999, epsilon=10e-8):
    """Single iteration of gradient descent using Adam.
    NB! In-place modification of 'v', 's' and 't'.
    
    Args:
        params (ndarray[1, n]):     parameters for current step
        grad (ndarray[1, n]):       gradient for current step
        alpha (float):              learning rate
        v (ndarray[1, n]):          1st order moment (exp. weighted average)
        s (ndarray[1, n]):          2nd order moment (exp. weighted average)
        t (int):                    iteration count for bias correction
        beta1 (float, optional):    1st order moment decay rate (defaults to 0.9)
        beta1 (float, optional):    2nd order moment decay rate (defaults to 0.999)
        epsilon (float, optional):  numerical stability constant (defaults to 10e-8)
    
    Returns:
        params (ndarray[1, n]):     updated parameters
    """
    t[...] = t + 1
    v[...] = beta1 * v + (1 - beta1) * grad
    s[...] = beta2 * s + (1 - beta2) * np.square(grad)
    v_corr = v / (1 - np.power(beta1, t))
    s_corr = s / (1 - np.power(beta2, t))
    return params - alpha * v_corr/(np.sqrt(s_corr) + epsilon)