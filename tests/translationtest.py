import matplotlib.pyplot as plt
from tests.testutility import TestUtility

if __name__ == "__main__":
    n = 5
    idb = TestUtility.get_n_random_imgs_idb(n)
    f, ax = plt.subplots(n, 2, sharey=True)

    imgs = idb.deserialize()
    translated_imgs = idb.get_translated_imgs(20, 40)

    for i in range(n):
        ax[i][0].imshow(imgs[i])
        ax[i][1].imshow(translated_imgs[i])
    plt.show()
