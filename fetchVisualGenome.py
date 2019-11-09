import requests
import json
import os
from PIL import Image
from io import BytesIO
from imageUtils import *

api = "http://visualgenome.org/api/v0/"
dirname = os.path.dirname(__file__)
objectsFilename = f"{dirname}/Data/VisualGenome/objects.json"

def getImagesByQuery(query, destination):
    file = open(objectsFilename)
    imageObjects = json.load(file)
    file.close()
    print("JSON LOADED")

    query = query.lower()
    queryResults = {}
    # queryResults will be a map of imageId : region Containing the largest region in each image that matches query
    # and is at least half targetDiminsions
    for img in imageObjects:
        largest = None
        for obj in img['objects']:
            if (any( (query in name) for name in obj['names'] ) and
                obj['w'] >= targetDims[0] // 2 and obj['h'] >= targetDims[1] // 2 and
                (largest == None or obj['w'] * obj['h'] > largest['w'] * largest['h'])
            ):
                largest = obj
        if (largest):
            queryResults[img['image_id']] = largest

    if not os.path.exists(destination):
        os.makedirs(destination)
    cocoDuplicateCount = 0
    for imageId, obj in queryResults.items():
        imageData = requests.get(api + f"images/{imageId}").json()
        if imageData['coco_id'] == None:
            image = Image.open( BytesIO( requests.get(imageData['url']).content ) )

            image = cropImageToBbox(image, (obj['x'], obj['y'], obj['w'], obj['h']) )
            image = resizeImage(image)

            image.save( os.path.join(destination, f"{imageId:08}.jpg") )
        else:
            cocoDuplicateCount += 1

    print("Done!")
    print(f"{len(queryResults) - cocoDuplicateCount} images saved. {cocoDuplicateCount} matching images also in COCO database not saved.")