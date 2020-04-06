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

# Batch
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