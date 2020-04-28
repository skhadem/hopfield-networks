import numpy as np
from PIL import Image
from distancemetrics import DistanceMetrics
from inputdatabuilder import InputDataBuilder
from tests.testutility import TestUtility
from hopfieldnetwork import HopfieldNetwork
import itertools
import os


def flatten_list_dict(d):
    ls = []
    for k, v in d.items():
            ls.append([(k, _) for _ in v])
    return ls


def map_to_tuple(x):
    if type(x) is tuple:
        return x

    return (x,)


class Experiment:
    EXPERIMENT_FPATH = "experiments"
    FUNCTION_MAP = {
        "snp": InputDataBuilder._add_salt_pepper_noise,
        "rot": InputDataBuilder._rotate_img,
        "trans": InputDataBuilder._translate_img,
        "stretch": InputDataBuilder._stretch_img,
    }

    def __init__(self, name, idb, noise_and_params=itertools.product([])):
        self.name = name
        self.idb = idb
        self.noise_and_params = noise_and_params

    def run(self):
        hn = HopfieldNetwork(np.prod(self.idb.size))
        hn.batch_update(self.idb.X)

        folder = "{}/{}".format(Experiment.EXPERIMENT_FPATH, self.name)
        try:
            os.mkdir(folder)
        except FileExistsError:
            pass

        np.save("{}/W.npy".format(folder), hn.W)

        with open("{}/params.txt".format(folder), 'w') as f:
            f.write(','.join(self.idb.fpaths) + '\n')

            for combination_noise_params in self.noise_and_params:
                f.write("{}\n".format(str(combination_noise_params)))

        for combination_noise_params in self.noise_and_params:
            imgs = self.idb.deserialize()
            spec = ""

            for noise, param in combination_noise_params:
                spec += "{}_{}".format(noise, '_'.join((str(_) for _ in param)))
                imgs = list(map(lambda x: Experiment.FUNCTION_MAP[noise](x, *param), imgs))

            for fpath, x in zip(self.idb.fpaths, self.idb.X):
                recalled_img = hn.recall(x)
                Image.fromarray(InputDataBuilder._deserialize(recalled_img, size=self.idb.size, display=True)) \
                    .save("{}/{}_recall_{}".format(folder, spec, fpath[29:]))


            for fpath, img in zip(self.idb.fpaths, imgs):
                Image.fromarray(255*img)\
                    .save("{}/{}_{}".format(folder, spec, fpath[29:]))


class ExperimentBuilder:
    IMAGES_FPATH = "data/MPEG7_CE-Shape-1_Part_B"
    DEFAULT_N = 5
    DEFAULT_IMGS = ["apple-1.gif", "turtle-9.gif", "ray-4.gif", "rat-19.gif", "personal_car-15.gif"]

    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.reset()

    def reset(self):
        self.n = ExperimentBuilder.DEFAULT_N
        self.idb = None
        self.noise_sources = {}
        self.params = []

    def with_noise(self, noise_source):
        for source, params in noise_source.items():
            if source in self.noise_sources:
                raise ValueError
            self.noise_sources[source] = [map_to_tuple(param) for param in params]
        self.params = list(itertools.product(*flatten_list_dict(self.noise_sources)))
        return self

    def with_n_random_images(self, n):
        self.n = n
        self.idb = TestUtility.get_n_random_imgs_idb(self.n)
        return self

    def with_imgs(self, fnames):
        self.n = None
        self.idb = InputDataBuilder(["{}/{}".format(ExperimentBuilder.IMAGES_FPATH, fname) for fname in fnames],
                                    size=self.size)
        return self

    def build(self):
        if self.idb is None:
            raise ValueError("No ImageDataBuilder provided")

        if self.params is []:
            raise ValueError("Empty experiment")

        return Experiment(self.name, self.idb, self.params)

