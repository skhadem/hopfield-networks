import matplotlib.pyplot as plt
from tests.testutility import TestUtility

if __name__ == "__main__":
    n = 5
    idb = TestUtility.get_n_random_imgs_idb(n)
    f, ax = plt.subplots(n, 2, sharey=True)

    imgs = idb.deserialize()
    rotated_imgs = idb.get_rotated_imgs(45)

    for i in range(len(n)):
        ax[i][0].imshow(imgs[i])
        ax[i][1].imshow(rotated_imgs[i])
    plt.show()
