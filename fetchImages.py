import sys
import fetchCOCO as COCO
import fetchOpenImages as OpenImages
import fetchPersonal as Personal
import fetchVisualGenome as VisualGenome

usage = "fetch.py category destination\n"

if __name__ == '__main__':
    args = sys.argv[1:]

    if (len(args) != 2):
        print(usage)
        sys.exit()

    category = args[0]
    destination = args[1].rstrip("/")

    COCO.fetchCategory(category, destination)
    OpenImages.fetchCategory(category, destination)
    Personal.fetchCategory(category, destination)
    VisualGenome.fetchCategory(category, destination)

    print(f"DONE! All images of {category} in all databases saved to {destination}")