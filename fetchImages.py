import sys
import fetchCOCO as COCO
import fetchOpenImages as OpenImages
import fetchPersonal as Personal
import fetchVisualGenome as VisualGenome

usage = "fetch.py destination categories...\n"

if __name__ == '__main__':
    args = sys.argv[1:]

    if (len(args) < 2):
        print(usage)
        sys.exit()

    destination = args[0].rstrip("/")
    categories = args[1:]

    for category in categories:
        COCO.fetchCategory(category, destination)
        OpenImages.fetchCategory(category, destination)
        Personal.fetchCategory(category, destination)
        VisualGenome.fetchCategory(category, destination)
        print("\n")

    print(f"DONE! All images of {', '.join(categories)} in all databases saved to {destination}")