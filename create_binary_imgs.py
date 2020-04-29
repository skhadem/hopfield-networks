import cv2
import numpy as np
from PIL import Image

x = np.array(Image.open("data/digits/0.png"))
x[5:7, 1:8] = 255
x[:, 1] = 0
x[:, 8] = 0
Image.fromarray(x).save("data/digits/8.png")

x = np.zeros(x.shape, dtype=x.dtype)
x[0:2, 1:-1] = 255
x[10:, 1:-1] = 255
x[2:7, 1:3] = 255
x[5:7, 1:-1] = 255
x[5:, 7:-1] = 255
Image.fromarray(x).save("data/digits/5.png")

x = np.zeros(x.shape, dtype=x.dtype)
x[0:2, 1:-1] = 255
x[2:4, 7:9] = 255
x[4:6, 6:8] = 255
x[6:8, 5:7] = 255
x[8:10, 4:6] = 255
x[10:, 3:5] = 255
Image.fromarray(x).save("data/digits/7.png")

print(x)

