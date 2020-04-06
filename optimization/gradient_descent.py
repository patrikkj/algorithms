# Batch

def _gradient_descent(x, func, grad_func, alpha=0.01):
    value, grad = func(x), grad_func(x)
    return x - alpha * grad, value, grad

def gradient_descent(x0, func, grad_func, alpha=0.01, num_iter=1000):
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


# Mini-batch
def mini_batch_gradient_descent(x0, func, grad_func, k=64, alpha=0.01, num_iter=1000):
    # Shuffle



    for batch in range(k):
        pass

# Stochastic
