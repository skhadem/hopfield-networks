import numpy as np
import matplotlib.pyplot as plt
import cv2

class HopfieldNetwork:
    def __init__(self, n):
        """
        Initialize Hopfield network
        :param n: number of neurons (length of input vector)
        """
        self.n = n
        self.W = np.zeros((n, n), dtype=np.float16)

    def update(self, p, strength=1, in_batch=False):
        """
        Update network weight matrix according to pattern p
        :param p: pattern for network to memorize
        :param in_batch: whether or not this update is occurring as part of a batch update
        """
        # flat array, need to make column vector
        if len(p.shape) == 1:
            p = p.reshape(len(p), 1)

        if in_batch:
            self.W += strength * np.matmul(p, p.T)
        else:
            self.W += (np.matmul(p, p.T)  - np.eye(self.n))

    def batch_update(self, patterns, strengths=[]):
        """
        Update network weight matrix according to multiple patterns
        :param patterns: patterns for network to memorize
        """

        if len(strengths) == 0:
            strengths = np.ones(len(patterns))

        for i,p in enumerate(patterns):
            self.update(p, strength=strengths[i], in_batch=True)

        self.W -= len(patterns) * np.eye(self.n)
        self.W /= self.n

    def recall(self, x, tol=1e-1, max_iter=100, verbose=False, save_evolution=0, evolution_name=None):
        """
        Get network recall output for input x
        :param x: input similar to one of stable state patterns network has been trained on
        :param tol: tolerance for fixed point convergence
        :param max_iter: max number of fixed point iterations
        :param save_evolution: pass in a size to save a gif to the data folder of the evolution of input
        :return: network recall output
        """
        mse = np.inf
        iter_ = -1
        images = []
        while mse >= tol and iter_ <= max_iter:
            if (save_evolution != 0):
                import imageio
                show_x = x.reshape((save_evolution, save_evolution))
                images.append(show_x)

            z = np.sign(np.matmul(self.W, x))
            z[z == 0] = -1
            mse = np.sum((z-x)**2)
            
            if verbose:
                print(mse, end='\n')
                
            x = z

            iter_ += 1

        if save_evolution:
            # make the last image linger more
            durations = [1 for i in range(0, len(images))]
            durations[-1] = 3
            imageio.mimsave(evolution_name, images, duration=durations)

        if verbose:
            if iter_ > max_iter:
                print("Hit max iters")
            if mse < tol:
                print("Hit MSE goal after %s iterations" % iter_)

        return x
