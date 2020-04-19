from PIL import Image
import numpy as np
import cv2


class InputDataBuilder:
    def __init__(self, fpaths, size=(100, 100)):
        """
        Ingest images and convert to input form for Hopfield network
        :param fpaths: filepaths of images to ingest
        :param size: target size for input question
        """
        self.fpaths = fpaths
        self.size = size
        self.X = []

        for fpath in self.fpaths:
            img = np.array(Image.open(fpath), dtype=np.uint8)               # cast array to uint8 for cv2::resize
            self.X.append(InputDataBuilder._serialize(img, self.size))

    @classmethod
    def _serialize(cls, img, size):
        """
        Convert image to input form for Hopfield network
        :param img: input image (dtype = unsigned 8-bit integer)
        :param size: target size of image
        :return: serialized image
        """
        x = np.array(np.sign(cv2.resize(img, size)), dtype=int)     # cast back to signed integer + store signs
        x[x == 0] = -1                                              # set all black pixels to -1 value
        return x.reshape((np.prod(size), 1))                        # convert to vector

    @classmethod
    def _deserialize(cls, x, size, display=False):
        """
        Convert data from input form for Hopfield network to image form readable by Pillow
        :param x: image in input form for Hopfield network
        :param size: original image dimensions
        :param display: whether or not to scale pixel values to 0-255 for display
        :return: deserialized image
        """
        if np.prod(size) != np.prod(x.shape):
            raise ValueError("Cannot reshape vector of shape {} to shape {}".format(x.shape, size))

        img = x.reshape(size)                               # reshape vector to image dimensions
        img[img == -1] = 0                                  # convert back to signed integer
        return np.array(img, dtype=np.uint8) if not display else np.array(img*255, dtype=np.uint8)

    def deserialize(self, display=False):
        """
        Convert images loaded into class from input form for Hopfield network to image form readable by Pillow
        :param display: whether or not to scale pixel values to 0-255 for display
        :return: array of deserialized images
        """
        imgs = []
        for x in self.X:
            imgs.append(InputDataBuilder._deserialize(x, self.size, display=display))

        return imgs

    @classmethod
    def _add_salt_pepper_noise(cls, img, p_salt=0.5, p_noise=0.25):
        """
        Add salt and pepper noise to an image
        :param img: target image
        :param p_salt: probability of salt (1 - probability of pepper)
        :param p_noise: probability of noise (for a specific pixel)
        :return: noisy image
        """
        noise = np.random.binomial(1, p_noise, img.shape)
        salt_pepper = np.random.binomial(1, p_salt, noise[noise==1].shape)
        noisy_img = img.copy()
        noisy_img[noise == 1] = salt_pepper
        return noisy_img

    def get_salt_pepper_noisy_imgs(self, p_salt=0.5, p_noise=0.25):
        """
        Get noisy versions of all images
        :param p_salt: probability of salt (1 - probability of pepper)
        :param p_noise: probability of noise (for a specific pixel)
        :return: noisy images
        """
        noisy_imgs = []
        for img in self.deserialize():
            noisy_imgs.append(InputDataBuilder._add_salt_pepper_noise(img, p_salt, p_noise))

        return noisy_imgs
