import numpy as np
import cv2
from matplotlib import pyplot as plt

def convolution_operation(surrondingPixels, kernel):
    sum = 0

    i = 0
    for pixelRow in surrondingPixels:
        j = 0
        for pixel in pixelRow:
            sum += pixel * kernel[i][j]
            j += 1
        i += 1
    
    return sum

def get_surronding_pixels(pixelMatrix, pixelPosition, size):
    w = size//2

    posX = pixelPosition[0]
    posY = pixelPosition[1]

    surrondingPixels = np.zeros((size,size))

    for i in range(size):
        relativeY = posY - w + i

        for j in range(size):
            relativeX = posX - w + j

            if (relativeX >= 0 and relativeX < pixelMatrix.shape[0]) and (relativeY >= 0 and relativeY < pixelMatrix.shape[0]):
                surrondingPixels[i][j] = pixelMatrix[relativeY][relativeX]        
            
    return surrondingPixels


def filter_2D(image, kernel, visualise):
    
    overallResult = np.array(image.shape)

    # Check for rgb or grey
    try:
        (redChannel, greenChannel, blueChannel ) = cv2.split(image)

        redResult = filter_2D_grey(redChannel, kernel)
        greenResult = filter_2D_grey(greenChannel, kernel)
        blueResult = filter_2D_grey(blueChannel, kernel)

        overallResult = cv2.merge((redResult, greenResult, blueResult))

    except:
        overallResult = filter_2D_grey(image, kernel)

    
    if visualise:
        visualise_change(image, overallResult)


    return overallResult

def filter_2D_grey(image, kernel):
    result = image.copy()
    
    for y in range(0, image.shape[0]):
        for x in range(0, image.shape[0]):
            surrondingPixels = get_surronding_pixels(image, (x, y), kernel.shape[0])
            pixelSum = convolution_operation(surrondingPixels, kernel)

            # Clamp value between 0 & 255
            if(pixelSum < 0):
                pixelSum = 0
            if(pixelSum > 255):
                pixelSum = 255

            result[y][x] = pixelSum    

    result = result.astype(np.uint8)
    return result


def visualise_change(before, after):

    inbetween = before.copy()

    for y in range(0, inbetween.shape[0]):
        for x in range(0, inbetween.shape[0]):

            inbetween[y][x] = after[y][x]

            if x%50 == 0:
                cv2.imshow('', inbetween)
                cv2.waitKey(1)