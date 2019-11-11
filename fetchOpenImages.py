import os
import csv
from PIL import Image
from imageUtils import *

dirname = os.path.dirname(__file__)
classFile = f"{dirname}/Data/OpenImages/class-descriptions-boxable.csv"
annotationsFiles = {
    "test"      : f"{dirname}/Data/OpenImages/test-annotations-bbox.csv",
    "train"     : f"{dirname}/Data/OpenImages/train-annotations-bbox.csv",
    "validation": f"{dirname}/Data/OpenImages/validation-annotations-bbox.csv"
}
imageMetadataFiles = {
    "test"      : f"{dirname}/Data/OpenImages/test-images-with-rotation.csv",
    "train"     : f"{dirname}/Data/OpenImages/train-images-boxable-with-rotation.csv",
    "validation": f"{dirname}/Data/OpenImages/validation-images-with-rotation.csv"
}

categories = {}
def getCategories():
    global categories
    if not categories:
        file = open(classFile)
        for line in csv.reader(file):
            categories[line[1].lower()] = line[0]
        file.close()
    return categories

def getImages():
    """ Returns a generator that returns tuples of (subset, imageId, [annotations]) """
    for subset, file in annotationsFiles.items():
        file = open(file)
        #    0   ,   1   ,     2    ,     3     ,  4     5     6     7        8            9          10          11          12
        # ImageID, Source, LabelName, Confidence, XMin, XMax, YMin, YMax, IsOccluded, IsTruncated, IsGroupOf, IsDepiction, IsInside
        reader = csv.reader(file); next(reader) # skip header

        # Annotations for the same image come in sequence. So yield the batches one at a time instead of loading
        # the whole thing in memory at once.
        currentImage = None
        currentAnns = []
        for line in reader:
            if (line[0] != currentImage):
                if (currentImage != None):
                    yield (subset, currentImage, currentAnns)
                currentImage = line[0]
                currentAnns = []
            currentAnns.append({
                "Source"     : line[1],
                "LabelName"  : line[2],
                "Confidence" : line[3],
                "xNorm"      : float(line[4]),
                "yNorm"      : float(line[6]),
                "wNorm"      : float(line[5]) - float(line[4]),
                "hNorm"      : float(line[7]) - float(line[6]),
                "IsOccluded" : bool(int(line[8])),
                "IsTruncated": bool(int(line[9])),
                "IsGroupOf"  : bool(int(line[10])),
                "IsDepiction": bool(int(line[11])),
                "IsInside"   : bool(int(line[12])),
            })
        yield (subset, currentImage, currentAnns)

imageRotations = {}
def getImageRotations():
    global imageRotations

    if not imageRotations:
        for file in imageMetadataFiles.values():
            file = open(file)
            #    0   ,   1   ,      2     ,          3        ,    4   ,         5       ,   6   ,   7  ,      8      ,      9     ,         10      ,    11
            # ImageID, Subset, OriginalURL, OriginalLandingURL, License, AuthorProfileURL, Author, Title, OriginalSize, OriginalMD5, Thumbnail300KURL, Rotation
            reader = csv.reader(file); next(reader) # skip header

            for line in reader:
                imageRotations[ line[0] ] = 0 if line[11] == "" else int(float(line[11]))
    return imageRotations

def batchImagesForCategory(categoryID, batchSize = 6):
    """ Generator that returns batches of [(subset, imageID, objectsInImage)] that match category. """
    imageRotations = getImageRotations()

    batch = [] # [(subset, imageID, objects)]
    for subset, imageID, anns in getImages():
        # get all objects matching category out of the image and crop each of them to their bbox.
        # Fetch them, and save each to a different image different image.
        # Exclude any objects that have bbox overlap with another.
        # Exclude any objects smaller than half targetSize
        objects = []
        for ann in anns:
            if (ann['LabelName'] == categoryID and
                not ann['IsDepiction'] and not ann['IsInside'] and not ann['IsGroupOf'] and
                imageRotations[imageID] == 0 # Not rotated
            ):
                for otherAnn in anns: # collision check
                    if (ann is not otherAnn and boxesOverlap(
                        (ann['xNorm'],      ann['yNorm'],      ann['wNorm'],      ann['hNorm']),
                        (otherAnn['xNorm'], otherAnn['yNorm'], otherAnn['wNorm'], otherAnn['hNorm'])
                    )):
                        break
                else: # no collisions
                    objects.append(ann)

        if objects:
            batch.append((subset, imageID, objects))

        if len(batch) >= batchSize:
            yield batch
            batch = []

    if batch:
        yield batch


def fetchImagesByCategory(category, destination):
    category = getCategories()[category.lower()]

    for batch in batchImagesForCategory(category, batchSize=8):
        command = [f'aws s3 --no-sign-request --only-show-errors cp s3://open-images-dataset/{img[0]}/{img[1]}.jpg "{destination}/" & ' for img in batch]
        os.system( ''.join(command) + "wait" ) # download
        for subset, imageId, objects in batch:
            # Crop objects out of image.
            img = Image.open( f"{destination}/{imageId}.jpg" )
            objIndex = 0
            for obj in objects:
                bbox = (
                    obj['xNorm'] * img.size[0], obj['yNorm'] * img.size[1],
                    obj['wNorm'] * img.size[0], obj['hNorm'] * img.size[1]
                )
                if bbox[2] >= targetDims[0] // 2 and bbox[3] >= targetDims[1] // 2:
                    croppedImage = cropImageToBbox(img, bbox)
                    resizedImage = resizeImage(croppedImage)
                    resizedImage.save( f"{destination}/{imageId}-sub{objIndex}.jpg" )
                    objIndex += 1

            img.close()
            os.remove(f"{destination}/{imageId}.jpg") # Remove original image