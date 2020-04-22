from tests.testutility import TestUtility
from hopfieldnetwork import HopfieldNetwork
from inputdatabuilder import InputDataBuilder
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    n = 5
    idb = TestUtility.get_n_random_imgs_idb(n, size=(50, 50))
    hn = HopfieldNetwork(50**2)
    hn.batch_update(idb.X)

    f, ax = plt.subplots(n, 1)

    for i, img in enumerate(idb.deserialize()):
        ax[i].imshow(img)

    plt.show()

    w, v = np.linalg.eig(hn.W)
    z = np.sign(np.real(v[np.argmin(abs(w-1))]))
    plt.imshow(InputDataBuilder._deserialize(z, size=(50, 50)), cmap='gray')
    plt.show()
