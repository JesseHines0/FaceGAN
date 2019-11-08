import json
import os

dirname = os.path.dirname(__file__)
annotationsFilename = f"{dirname}/Data/COCO/annotations-trainval/instances_train2017.json"

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
    catMap = getCategories()
    catsToFetch = []

    if category in catMap: # Supercategory
        catsToFetch = [cat['id'] for cat in catMap[category]]
    else:
        for superCat in catMap: # find category
            matching = list( filter(lambda t: t[1] == category, catMap[superCat]) )
            if matching:
                catsToFetch = [matching[0][0]]
                break
    
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
