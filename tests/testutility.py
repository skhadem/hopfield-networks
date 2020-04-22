import random
from inputdatabuilder import InputDataBuilder
import os

class TestUtility:
    SHAPES_PATH = "../data/MPEG7_CE-Shape-1_Part_B"

    @classmethod
    def get_n_random_imgs_idb(cls, n, **kwargs):
        names = [_ for _ in os.listdir(cls.SHAPES_PATH) if _[-4:] == '.gif']
        random.shuffle(names)
        fpaths = ["{}/{}".format(cls.SHAPES_PATH, name) for name in names[:n]]
        return InputDataBuilder(fpaths, **kwargs)
