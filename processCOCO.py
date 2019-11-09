import json
import os
from PIL import Image

dirname = os.path.dirname(__file__)
annotationsFilename = f"{dirname}/Data/COCO/annotations-trainval/instances_train2017.json"
imagesDir = f"{dirname}/Data/COCO/images-train/"
targetDims = (336, 280)

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
    """ returns a list containing the id of the category, or a list containing the ids of all subcategories """
    catMap = getCategories()
    if category in catMap: # Supercategory
        return [cat['id'] for cat in catMap[category]]
    else:
        for superCat in catMap: # find category
            matching = list( filter(lambda t: t[1] == category, catMap[superCat]) )
            if matching:
                return [matching[0][0]]
            
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

def getImagesInCategory(category, filterImages = False):
    """
    Returns a list of file objects for all images matching category or one of its sub-categories in the COCO dataset.
    category is the name of a category. If filter is true, it will remove images that have crowds or more than one category.
    """
    catsToFetch = getCategoryIds(category)
    
    # fetch images for category
    allImages = getImages()
    imagesInCat = []
    imageAnnotations = getImageAnnotations()

    for image_id, annotations in imageAnnotations.items():
        # filter out crowd images or images with multiple categories.
        if (not filterImages) or (len(annotations) == 1 and annotations[0]['iscrowd'] == 0):
            for annotation in annotations: # check if the categories match
                if annotation['category_id'] in catsToFetch:
                    imagesInCat.append( allImages[ image_id ] )
                    break

    return imagesInCat

def boxesOverlap(a, b):
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


def processImagesForCategory(category, destination):
    catsToFetch = getCategoryIds(category)
    images = getImages()
    imageAnns = getImageAnnotations()

    if not os.path.exists(destination):
        os.makedirs(destination)
        
    for image_index, (image_id, anns) in enumerate(imageAnns.items()):
        if (image_index % 1000 == 0):
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
                for otherAnn in anns: # collision check
                    if ann['id'] != otherAnn['id'] and boxesOverlap(ann['bbox'], otherAnn['bbox']):
                        break
                else: # no collisions
                    objects.append(ann)
        
        img = Image.open( imagesDir + images[ image_id ] )
        for objIndex, obj in enumerate(objects):
            bbox = obj['bbox']
            croppedImage = img.crop( (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]) )

            resizedImage = resizeImage(croppedImage)

            sub = chr(97 + objIndex) if objIndex < 26 else f"sub{objIndex}"
            resizedImage.save( os.path.join(destination, f"{images[image_id].rsplit('.', 1)[0]}{sub}.jpg") )
           
    print("DONE!")

