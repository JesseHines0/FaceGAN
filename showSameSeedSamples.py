import os
import matplotlib.pyplot as plt
from PIL import Image
import math

def showImages():
    """ Displays images in sample_images/celebA64-v2-sameSeed in order of the checkpoint that made them. """
    groups = {}
    for file in os.listdir("sample_images/celebA64-v2-sameSeed"):
        seed = file.split("_")[-1].split(".")[0]
        if seed not in groups:
            groups[seed] = []
        groups[seed].append(file)

    for group in groups.values():
        group.sort(key=lambda file: int(file.split("_")[2].split("-")[1]) )

    groups = list(groups.items())
    groups.sort(key= lambda e: int(e[0]))


    display_width = math.ceil(math.sqrt(len(groups[0][1])))
    display_height = math.ceil( len(groups[0][1]) / display_width )

    images = []
    axes = []
    for i in range(len(groups[0][1])):
        axis = plt.subplot(display_width, display_height, i+1)
        axes.append(axis)

        fig = plt.imshow([[[0, 0, 0]]], interpolation="none")
        images.append( fig )

        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

    for seed, group in groups:
        if not plt.get_fignums(): return

        for i, file in enumerate(group):

            image = Image.open(f"sample_images/celebA64-v2-sameSeed/{file}")
            images[i].set_data(image)

            epoch = file.split("_")[2].split("-")[1]
            axes[i].set_xlabel(epoch)

        plt.pause(10)

showImages()
