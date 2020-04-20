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
            print(self.W)

    def store_information(self, pattern):
        # TODO: naive implemenation, super slow. Need to convert to matrix mults
        for i in range(0, self.n):
            for j in range(0, i):
                if (i != j):
                    weight = pattern[i]*pattern[j]
                    self.W[i,j] = weight
                    self.W[j,i] = weight


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
        output_pattern = np.zeros(self.n)
        for i in range(0, self.n):
            if np.sum(self.W[i,:]*x) > tol:
                output_pattern[i] = 1
            else:
                output_pattern[i] = -1

        return output_pattern
