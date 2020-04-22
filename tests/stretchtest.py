from tests.testutility import TestUtility
import matplotlib.pyplot as plt

if __name__ == "__main__":
    n = 5
    idb = TestUtility.get_n_random_imgs_idb(n)
    f, ax = plt.subplots(n, 3, sharey=True)

    imgs = idb.deserialize()
    stretched_imgs = idb.get_stretched_imgs(1.5, 1.5)
    shrunk_imgs = idb.get_stretched_imgs(0.6, 0.6)

    for i in range(n):
        ax[i][0].imshow(imgs[i])
        ax[i][1].imshow(stretched_imgs[i])
        ax[i][2].imshow(shrunk_imgs[i])
    plt.show()
