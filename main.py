import numpy as np
import cv2
from matplotlib import pyplot as plt
import convolution as conv

H_SOBEL_5X5 = np.array([
    [ 2,  2,  4,  2,  2],
    [ 1,  1,  2,  1,  1],
    [ 0,  0,  0,  0,  0],
    [-1, -1, -2, -1, -1],
    [-2, -2, -4, -2, -2],
])

V_SOBEL_5X5 = np.array([
    [2, 1, 0, -1, -2],
    [2, 1, 0, -1, -2],
    [4, 2, 0, -2, -4],
    [2, 1, 0, -1, -2],
    [2, 1, 0, -1, -2],
])

H_SOBEL_3X3 = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1],
])

V_SOBEL_3X3 = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1],
])

NO_EFFECT = np.array([
    [0, 0, 0],
    [0, 2, 0],
    [0, 0, 0],
    ])

#### START #####
image = cv2.imread('test.png')
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# image = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9],
# ], dtype=np.uint8)


# cv2.imshow('',image)

kernel = cv2.flip(H_SOBEL_3X3, -1)
kernelStr = 'Sobel 3x3 Horizontal'

# theirFilter = image.copy()
theirFilter = cv2.filter2D(image.copy(), -1,  kernel)
myFilter = conv.filter_2D(image.copy(), kernel, True)

plt.subplot(2,2,1),plt.imshow(image,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,2),plt.imshow(theirFilter,cmap = 'gray')
plt.title('Theirs - ' + kernelStr), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,3),plt.imshow(myFilter,cmap = 'gray')
plt.title('Mine - ' + kernelStr), plt.xticks([]), plt.yticks([])


plt.show()