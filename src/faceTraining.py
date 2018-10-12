import os
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

x_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            # get path to images
            path = os.path.join(root, file)
            # get path to folders in images which are labels to the respective images
            label = os.path.basename(os.path.dirname(path)).lower()
            # print(label, path)
            pil_image = Image.open(path).convert("L")  # to grayscale read pillow documentation
            image_array = np.array(pil_image, "uint8")  # convert image to numpy array values
