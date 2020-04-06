import numpy as np


def mini_batch_gradient_descent(params, X, y, cost_func, grad_func, alpha=0.01, epochs=100, k=64):
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
            grad = grad_func(params, X_batch, y_batch)
            cost = cost_func(params, X_batch, y_batch)
            params = params - alpha * grad_func(params, X_batch, y_batch)
            #print("X=", X_batch)
            #print("y=", y_batch)
            #print("params=", params)
            #print("cost=", cost)
            #print("grad=", grad)
            #print()
            costs.append(cost)
            grads.append(grad)
    return params, costs, grads

def gradient_descent(params, X, y, cost_func, grad_func, alpha=0.01, epochs=100):
    return mini_batch_gradient_descent(params, X, y, cost_func, grad_func, alpha, epochs=epochs, k=X.shape[0])

def stochastic_gradient_descent(params, X, y, cost_func, grad_func, alpha=0.01, epochs=100):
    return mini_batch_gradient_descent(params, X, y, cost_func, grad_func, alpha, epochs, 1)
