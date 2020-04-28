import numpy as np


class DistanceMetrics:
    @classmethod
    def overlap(cls, normed=False, *args):
        """
        Overlap between n vectors specified in args
        :param normed: whether or not to normalize distance
        :param args: vectors to calculate overlap for
        :return:
        """
        args = list(args)
        x = args.pop()
        acc = np.zeros(x.shape) == 0
        while args:
            acc = x != args.pop()
        dist = np.sum(acc)

        return dist / np.prod(x.shape) if normed else dist
