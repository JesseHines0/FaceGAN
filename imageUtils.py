import os
from PIL import Image

targetDims = (336, 280)

def boxesOverlap(a, b):
    """ Returns true if two (x, y, width, height) bounding boxes overlap """
    return (
        # (x, y, width, height)
        # Dist between    < width/height of first box.
        (abs(a[0] - b[0]) < min(a, b, key = lambda box: box[0])[2]) and
        (abs(a[1] - b[1]) < min(a, b, key = lambda box: box[1])[3])
    )

def resizeImage(image):
    """ Resizes the image to targetDims, leaving padding where needed. Returns resized image."""
    ratioX = targetDims[0] / image.size[0]
    ratioY = targetDims[1] / image.size[1]
    ratio  = min(ratioX, ratioY)

    newSize = ( int(ratio * image.size[0]), int(ratio * image.size[1]) )

    # thumbnail is inplace
    resizedImage = image.resize(newSize, Image.ANTIALIAS)

    # create a new image and paste the resized on it
    paddedImage = Image.new("RGB", targetDims)
    paddedImage.paste(resizedImage, (
        (targetDims[0] - newSize[0]) // 2,
        (targetDims[1] - newSize[1]) // 2
    ))

    return paddedImage

def cropImageToBbox(image, bbox):
    """ Crops image to (x, y, width, height) bbox """
    return image.crop( (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]) )
