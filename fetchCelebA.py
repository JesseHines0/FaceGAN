import os
from PIL import Image
from imageUtils import *
import sys
import getopt

dirname = os.path.dirname(__file__)
annotationsFilename = f"{dirname}/Data/CelebA/Anno/list_bbox_celeba_abridged.txt"

imagesDir = f"{dirname}/Data/CelebA/img_align_celeba/"

def fetch(targetDims, destination):

    # with open(annotationsFilename) as file:
    #     if not os.path.exists(destination):
    #         os.makedirs(destination)

    #     # Skip headers.
    #     imageCount = int( file.readline() )
    #     file.readline()

    #     for index, line in enumerate(file):
    #         if (index % 5000 == 0):
    #             print(f"Processing image {index} of {imageCount}...")

    #         filename, *bbox = line.split()
    #         bbox = [int(x) for x in bbox]

    #         image = Image.open( imagesDir + filename )
    #         image = cropImageToBbox(image, bbox, targetDims)
    #         image.save( f"{destination}/{filename}" )

    if not os.path.exists(destination):
        os.makedirs(destination)

    for index, filename in enumerate(os.listdir(imagesDir)):
        image = Image.open( imagesDir + filename )
        image = image.resize(targetDims, Image.ANTIALIAS)

        image.save( f"{destination}/{filename}" )

    print(f"DONE! Processed all CelebA images.")