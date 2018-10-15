import cv2
import sys
import pickle

# Get user supplied values
imagePath = sys.argv[1]

#if fetching from file
cascPath = "../cascades/data/haarcascade_frontalface_default.xml" #sys.argv[2]
# if fetching via pip
# faceCascade =cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create the haar cascade path and link to classifier
faceCascade = cv2.CascadeClassifier(cascPath)

recognizer = cv2.face.LBPHFaceRecognizer_create()  #face recognizer module
recognizer.read("./recognizers/face-trainer.yml")

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# create label dictionary from pickle labels
labels = {"person": 1}
with open("pickles/labels.pickle", "rb") as f:
    first_labels = pickle.load(f)
    labels = { v:k for k,v in first_labels.items()}

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:

    roi_gray = gray[y:y + h, x:x + w]
    id_, conf = recognizer.predict(roi_gray)
    if conf >= 45 and conf <= 85:
        # print(id_)
        # print(labels[id_])
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 1
        name = labels[id_]
        color = (255, 255, 255)
        stroke = 2
        cv2.putText(image, name, (x, y), font, font_size, color, stroke, cv2.LINE_AA)


    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)