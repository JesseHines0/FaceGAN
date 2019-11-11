import os
from imageUtils import *
from PIL import Image

dirname = os.path.dirname(__file__)

def fetchImagesByCategory(category, destination):
    folder = f"{dirname}/Data/Personal/{category.lower()}"

    if not os.path.exists(destination):
        os.makedirs(destination)

    if os.path.exists(folder):
        for file in os.listdir(folder):
            img = Image.open(f"{folder}/{file}")
            resizedImage = resizeImage(img)
            resizedImage.save(f"{destination}/{file}")
            img.close()
