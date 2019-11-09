import requests
import json
import os
from PIL import Image
from io import BytesIO
from imageUtils import *

api = "http://visualgenome.org/api/v0/"
dirname = os.path.dirname(__file__)
regionsFilename = f"{dirname}/Data/VisualGenome/region_descriptions.json"

def getAllImageIds():
    allIds = []
    nextPage = api + "images/all"
    while (nextPage):
        result = requests.get(nextPage).json()
        allIds.extend(result["results"])
        nextPage = result["next"]
        print(nextPage)
    return allIds

def getImagesByQuery(query, destination):
    file = open(regionsFilename)
    imageRegions = json.load(file)
    print("JSON LOADED")

    query = query.lower()
    queryResults = {}
    # queryResults will be a map of imageId : region Containing the largest region in each image that matches query 
    # and is at least half targetDiminsions
    for img in imageRegions: 
        for region in img['regions']:
            largest = None
            if (query in region['phrase'].lower() and
                region['width'] >= targetDims[0] // 2 and region['height'] >= targetDims[1] // 2 and 
                (largest == None or region['width'] * region['height'] > largest['width'] * largest['height'])
            ):
                largest = region
        if (largest):
            queryResults[img['id']] = largest

    if not os.path.exists(destination):
        os.makedirs(destination)
    cocoDuplicateCount = 0
    for imageId, region in queryResults.items():
        imageData = requests.get(api + f"images/{imageId}").json()
        if imageData['coco_id'] == None:
            image = Image.open( BytesIO( requests.get(imageData['url']).content ) )
            image.save( os.path.join(destination, f"{imageId:08}.jpg") )
        else:
            cocoDuplicateCount += 1

    print("Done!")
    print(f"{len(queryResults) - cocoDuplicateCount} images saved. {cocoDuplicateCount} matching images also in COCO database not saved.")