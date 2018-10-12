import os
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade =cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            # get path to images
            path = os.path.join(root, file)
            # get path to folders in images which are labels to the respective images
            label = os.path.basename(os.path.dirname(path)).lower()
            # print(label, path)

            # check if labels have been given ids so as to append to y_labels by id
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]

            pil_image = Image.open(path).convert("L")  # to grayscale read pillow documentation
            image_array = np.array(pil_image, "uint8")  # convert image to numpy array values
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_array[y:y + h, x:x + w]
                x_train.append(roi)
                y_labels.append(id_)