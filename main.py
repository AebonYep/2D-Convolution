
import cv2
from matplotlib import pyplot as plt
import convolution as conv
import kernels

#### START #####
original = cv2.imread('test.png')
imageRGB = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
imageGrey = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)

kernel = cv2.flip(kernels.H_SOBEL_5X5, -1)
kernelStr = 'Horizontal Sobel 5x5'

# theirFilter = image.copy()
theirFilterRGB = cv2.filter2D(imageRGB.copy(), -1,  kernel)
myFilterRGB = conv.filter_2D(imageRGB.copy(), kernel, False)

theirFilterGrey = cv2.filter2D(imageGrey.copy(), -1,  kernel)
myFilterGrey = conv.filter_2D(imageGrey.copy(), kernel, False)


plotHeight = 3
plotWidth = 2

# RGB Plots
plt.subplot(plotHeight, plotWidth, 1),plt.imshow(imageRGB, cmap='gray')
plt.title('RBG - Original'), plt.xticks([]), plt.yticks([])

plt.subplot(plotHeight, plotWidth,3),plt.imshow(theirFilterRGB, cmap='gray')
plt.title('cv2.filter2D() - ' + kernelStr), plt.xticks([]), plt.yticks([])

plt.subplot(plotHeight, plotWidth,5),plt.imshow(myFilterRGB, cmap='gray')
plt.title('Mine - ' + kernelStr), plt.xticks([]), plt.yticks([])

# Greyscale plots
plt.subplot(plotHeight, plotWidth, 2),plt.imshow(imageGrey, cmap='gray')
plt.title('Greyscale - Original'), plt.xticks([]), plt.yticks([])

plt.subplot(plotHeight, plotWidth,4),plt.imshow(theirFilterGrey, cmap='gray')
plt.title('cv2.filter2D() - ' + kernelStr), plt.xticks([]), plt.yticks([])

plt.subplot(plotHeight, plotWidth,6),plt.imshow(myFilterGrey, cmap='gray')
plt.title('Mine - ' + kernelStr), plt.xticks([]), plt.yticks([])




plt.show()