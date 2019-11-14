import sys
import fetchCOCO as COCO
import fetchOpenImages as OpenImages
import fetchPersonal as Personal
import fetchVisualGenome as VisualGenome
import getopt


usage = ("fetch.py destination categories...\n" +
        "    Places images for all categories, resized and cropped, to destination. Each category will be under a folder of the same name.\n")

if __name__ == '__main__':
    args = sys.argv[1:]
    try:
        opts, args = getopt.getopt(args, 'r:', ['resolution='])
    except getopt.GetoptError as err:
        print(err)
        print(usage)
        sys.exit(2)

    if (len(args) < 2):
        print(usage)
        sys.exit()

    targetDims = (336, 280)
    try:
        for opt, val in opts:
            if opt in ['-r', '--resolution']:
                targetDims = val.split('x', 1)
                targetDims = (int(targetDims[0]), int(targetDims[1]))
    except Exception as err:
        print(err)
        print(usage)
        sys.exit(2)

    destination = args[0].rstrip("/")
    categories = args[1:]

    for category in categories:
        COCO.fetchCategory(category, targetDims, f"{destination}/{category}")
        OpenImages.fetchCategory(category, targetDims, f"{destination}/{category}")
        Personal.fetchCategory(category, targetDims, f"{destination}/{category}")
        VisualGenome.fetchCategory(category, targetDims, f"{destination}/{category}")
        print("\n")

    print(f"DONE! All images of {', '.join(categories)} in all databases saved to {destination}")