import os

dirname = os.path.dirname(__file__)

def fetchImagesForClass(classname):
    classname = classname.replace(' ', '_')
    os.chdir(f"{dirname}/OIDv4_ToolKit") # Change working directory

    # Download images with bounding boxes.
    os.system(f"python3 main.py downloader -y --classes {classname} --type_csv all --n_threads 3 " +
               "--image_IsOccluded 0 --image_IsTruncated 0 --image_IsGroupOf 0 --image_IsDepiction 0 --image_IsInside 0" # filter unusual images.
    )
    # Download images without bounding boxes.
    os.system(f"python3 main.py downloader_ill -y --sub h --classes {classname} --type_csv all --n_threads 3")


