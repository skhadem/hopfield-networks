import os
from inputdatabuilder import InputDataBuilder
import random
import matplotlib.pyplot as plt

if __name__ == "__main__":
    n = 5
    path = "../data/MPEG7_CE-Shape-1_Part_B"
    names = [_ for _ in os.listdir(path) if _[-4:] == '.gif']
    random.shuffle(names)
    fpaths = ["{}/{}".format(path, name) for name in names[:n]]
    idb = InputDataBuilder(fpaths)
    f, ax = plt.subplots(n, 3, sharey=True)

    imgs = idb.deserialize()
    stretched_imgs = idb.get_stretched_imgs(1.5, 1.5)
    shrunk_imgs = idb.get_stretched_imgs(0.6, 0.6)

    for i in range(len(fpaths)):
        ax[i][0].imshow(imgs[i])
        ax[i][1].imshow(stretched_imgs[i])
        ax[i][2].imshow(shrunk_imgs[i])
    plt.show()
