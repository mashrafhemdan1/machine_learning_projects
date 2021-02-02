import numpy as np

### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    kernel_matrix = np.power((np.matmul(X, np.transpose(Y)) + c), p)
    return kernel_matrix
    raise NotImplementedError

def EDM(A, B):
    p1 = np.sum(A**2, axis = 1)[:, np.newaxis]
    p2 = np.sum(B**2, axis = 1)
    p3 = -2*np.dot(A, B.T)
    return p1+p2+p3

def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    p1 = np.sum(X ** 2, axis=1)[:, np.newaxis]
    p2 = np.sum(Y ** 2, axis=1)
    p3 = -2 * np.dot(X, Y.T)
    kernel_matrix = np.exp(-gamma*(p1+p2+p3))
    return kernel_matrix
    raise NotImplementedError
