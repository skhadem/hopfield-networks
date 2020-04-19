import numpy as np


class HopfieldNetwork:
    def __init__(self, n):
        """
        Initialize Hopfield network
        :param n: number of neurons (length of input vector)
        """
        self.n = n
        self.W = np.zeros((n, n))

    def update(self, p, in_batch=False):
        """
        Update network weight matrix according to pattern p
        :param p: pattern for network to memorize
        :param in_batch: whether or not this update is occurring as part of a batch update
        """
        if in_batch:
            self.W += np.matmul(p, p.T)
        else:
            self.W += (np.matmul(p, p.T) - np.eye(self.n))

    def batch_update(self, patterns):
        """
        Update network weight matrix according to multiple patterns
        :param patterns: patterns for network to memorize
        """
        for p in patterns:
            self.update(p, in_batch=True)

        self.W -= len(patterns) * np.eye(self.n)
        self.W /= self.n

    def recall(self, x, tol=1e-1):
        """
        Get network recall output for input x
        :param x: input similar to one of stable state patterns network has been trained on
        :param tol: tolerance for fixed point convergence
        :return: network recall output
        """
        # TODO: implement network evolution asynchronously
        pass
