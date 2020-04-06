# Batch
import numpy as np

def _gradient_descent(x, cost, grad, alpha=0.01):
    """Single iteration of gradient descent.
    
    Args:
        x (float, [m, ...]): parameter to optimize (m, ...)
        func (x -> float): cost function
        grad_func (x -> float): gradient function
        alpha (float, optional): learning rate. Defaults to 0.01.
    
    Returns:
        [type]: [description]
    """
    return x - alpha * grad

def gradient_descent_(x0, func, grad_func, alpha=0.01, num_iter=1000):
    '''Minimizes the objective function using Gradient Descent.

    Arguments:
        x0:         initial parameters
        func:       objective function
        grad_func:  function of the form 'x0 -> gradient'

        (Optional)
        alpha:      learning rate
        num_iter:   number of iterations
    
    Returns:
        x:          optimized parameters
        values:     optimization objective evaluated for each iteration
        grads:      list of gradients for each iteration
    '''
    values, grads = [], []
    for i in range(num_iter):
        x0, value, grad = _gradient_descent(x0, func, grad_func, alpha)
        values.append(value)
        grads.append(grad)
    return x0, values, grads



def gradient_descent(params, X, Y, cost_func, grad_func, alpha=0.01, num_iter=1000, cache=True):
    costs, grads = [], []
    for i in range(num_iter):
        # Step
        cost, grad = cost_func(X, Y), grad_func(X, Y)
        params = params - alpha * grad

        # Cache values
        if cache:
            costs.append(cost)
            grads.append(grad)
    return params, costs, grads


# Mini-batch
def mini_batch_gradient_descent(params, X, Y, cost_func, grad_func, k=64, alpha=0.01, num_iter=1000):
    # Shuffle input
    X = np.random.shuffle(X)
    
    # Partition input
    m = X.shape[0]
    X_batches = np.split(X, range(0, m, k))
    Y_batches = np.split(Y, range(0, m, k))
    for X_batch, Y_batch in zip(X_batches, Y_batches):
        output = gradient_descent(params, X_batch, Y_batch, cost_func, grad_func, alpha, num_iter=1, cache=False)
        params, costs, grads = output


    for batch in range(k):
        pass

# Stochastic
