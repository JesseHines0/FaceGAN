import json
import os
from PIL import Image
from imageUtils import *

dirname = os.path.dirname(__file__)
annotationsFilename = f"{dirname}/Data/COCO/annotations-trainval/instances_train2017.json"
imagesDir = f"{dirname}/Data/COCO/images-train/"

fetchedMap = None
def getData():
    """ Loads annotations from file into a dictionary. Caches the result. """
    global fetchedMap
    if not fetchedMap:
        file = open(annotationsFilename)
        fetchedMap = json.load(file)
        print("JSON LOADED")
        file.close()
    return fetchedMap

def getCategories():
    """ Gets all categories in a dict of {supercategory: [subcategories]}. Categories are only nested one layer. """
    categories = getData()['categories']
    catMap = {}
    for cat in categories:
        if cat['supercategory'] not in catMap:
            catMap[cat['supercategory']] = []
        catMap[cat['supercategory']].append( (cat['id'], cat['name']) )

    return catMap

def getCategoryIds(category):
    """ returns a list containing the id of the category, or a list containing the ids of all subcategories.
        Returns empty lists if no matching category. """
    catMap = getCategories()
    if category in catMap: # Supercategory
        return [cat['id'] for cat in catMap[category]]
    else:
        for superCat in catMap: # find category
            matching = list( filter(lambda t: t[1] == category, catMap[superCat]) )
            if matching:
                return [matching[0][0]]
    return []

def getImages():
    """ Returns the images as a map of "id" to "filename" """
    imagesMap = {}
    images = getData()['images']
    for image in images:
        imagesMap[image['id']] = image['file_name']
    return imagesMap

def getImageAnnotations():
    """ Returns a map of image_id : [annotations] """
    annotations = getData()["annotations"]
    imagesMap = {}
    for annotation in annotations:
        if annotation['image_id'] not in imagesMap:
            imagesMap[ annotation['image_id'] ] = []
        imagesMap[ annotation['image_id'] ].append(annotation)
    return imagesMap

def fetchCategory(category, targetDims, destination):
    catsToFetch = getCategoryIds(category)
    if (catsToFetch == []):
        print(f"No COCO images for {category}.")
        return

    images = getImages()
    imageAnns = getImageAnnotations()

    if not os.path.exists(destination):
        os.makedirs(destination)

    for image_index, (image_id, anns) in enumerate(imageAnns.items()):
        if (image_index % 5000 == 0):
            print(f"Processing image {image_index} of {len(images)}...")
        # get all objects matching category out of the image and crop each of them to their bbox.
        # Saved each to different image.
        # Exclude any objects that have bbox overlap with another.
        # Exclude any objects smaller than half targetSize
        # Expand bbox to be closest to the aspect ratio of targetDims possible without bbox collision.
        objects = []
        for ann in anns:
            if (
                ann['category_id'] in catsToFetch and
                ann['iscrowd'] == 0 and
                ann['bbox'][2] >= targetDims[0] // 2 and ann['bbox'][3] >= targetDims[1] // 2
            ):
                # for otherAnn in anns: # collision check
                #     if ann['id'] != otherAnn['id'] and boxesOverlap(ann['bbox'], otherAnn['bbox']):
                #         break
                # else: # no collisions
                objects.append(ann)

        img = Image.open( imagesDir + images[ image_id ] )
        for objIndex, obj in enumerate(objects):
            croppedImage = cropImageToBbox(img, obj['bbox'])

            resizedImage = resizeImage(croppedImage, targetDims)

            sub = chr(97 + objIndex) if objIndex < 26 else f"sub{objIndex}"
            resizedImage.save( f"{destination}/COCO-{images[image_id].rsplit('.', 1)[0]}{sub}.jpg" )

    print(f"DONE! Processed all images of {category} in COCO database.")

