import os
from PIL import Image

def boxesOverlap(a, b):
    """ Returns true if two (x, y, width, height) bounding boxes overlap """
    return (
        # (x, y, width, height)
        # Dist between    < width/height of first box.
        (abs(a[0] - b[0]) < min(a, b, key = lambda box: box[0])[2]) and
        (abs(a[1] - b[1]) < min(a, b, key = lambda box: box[1])[3])
    )

def resizeImage(image, targetDims):
    """ Resizes the image to targetDims, leaving padding where needed. Returns resized image."""
    return image.resize(targetDims, Image.ANTIALIAS)

    # ratioX = targetDims[0] / image.size[0]
    # ratioY = targetDims[1] / image.size[1]
    # ratio  = min(ratioX, ratioY)

    # newSize = ( int(ratio * image.size[0]), int(ratio * image.size[1]) )

    # # thumbnail is inplace
    # resizedImage = image.resize(newSize, Image.ANTIALIAS)

    # # create a new image and paste the resized on it
    # paddedImage = Image.new("RGB", targetDims)
    # paddedImage.paste(resizedImage, (
    #     (targetDims[0] - newSize[0]) // 2,
    #     (targetDims[1] - newSize[1]) // 2
    # ))

    # return paddedImage

def _expandToFit(imageSize, newSize, origSize, origPos):
    """
    Returns the (newPos, newSize) in a single diminsion of a resized bbox, centering it
    and not going out of bounds of the image.
    """
    if newSize > imageSize: # shrink to fit in image.
        return (0, imageSize) # just take up all the space in the image that we have.
    else:
        # update x to have the old bbox in about middle of the new, accounting for image boundaries
        newPos = origPos - max(
            (origPos + newSize) - imageSize, # overlap on the right
            min(
                origPos, # dist to left side
                0.5 * (newSize - origSize) # half our change in width (centered)
            )
        )
        return (newPos, newSize)

def cropImageToBbox(image, bbox, ratio=None):
    """
    Crops image to (x, y, width, height) bbox.
    If ratio is given as a tuple of (width, height), will expand bbox to be the same ratio if
    it can given the size of the image.
    """
    newBbox = list(bbox)
    if ratio: # expand bbox.
        ratio = ratio[0] / ratio[1] # width / height
        if (bbox[2]/ bbox[3] < ratio):
            newSize = ratio * bbox[3] #expand width to match ratio
            newBbox[0], newBbox[2] = _expandToFit(image.size[0], newSize, bbox[2], bbox[0])
        else:
            newSize = bbox[2] / ratio # expand height to match ratio
            newBbox[1], newBbox[3] = _expandToFit(image.size[1], newSize, bbox[3], bbox[1])

    return image.crop( (newBbox[0], newBbox[1], newBbox[0] + newBbox[2], newBbox[1] + newBbox[3]) )
