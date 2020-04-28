import numpy as np


class HopfieldNetwork:
    def __init__(self, n):
        """
        Initialize Hopfield network
        :param n: number of neurons (length of input vector)
        """
        self.n = n
        self.W = np.zeros((n, n), dtype=np.float16)

    def update(self, p, in_batch=False):
        """
        Update network weight matrix according to pattern p
        :param p: pattern for network to memorize
        :param in_batch: whether or not this update is occurring as part of a batch update
        """
        # flat array, need to make column vector
        if len(p.shape) == 1:
            p = p.reshape(len(p), 1)

        if in_batch:
            self.W += np.matmul(p, p.T)
        else:
            self.W += (np.matmul(p, p.T)  - np.eye(self.n))

    def batch_update(self, patterns):
        """
        Update network weight matrix according to multiple patterns
        :param patterns: patterns for network to memorize
        """
        for p in patterns:
            self.update(p, in_batch=True)

        self.W -= len(patterns) * np.eye(self.n)
        self.W /= len(patterns)

    def recall(self, x, tol=1e-3, max_iter=100):
        """
        Get network recall output for input x
        :param x: input similar to one of stable state patterns network has been trained on
        :param tol: tolerance for fixed point convergence
        :param max_iter: max number of fixed point iterations
        :return: network recall output
        """

        mse = np.inf
        iter_ = 0
        while mse >= tol and iter_ <= max_iter:
            print(mse, end='\r')
            z = np.sign(np.matmul(self.W, x))
            z[z == 0] = 1
            mse = np.sum((z-x)**2)
            x = z

            iter_ += 1

        if iter_ > max_iter:
            print("Hit max iters")
        if mse < tol:
            print("Hit MSE goal")

        return x
