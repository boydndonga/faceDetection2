import cv2
import pickle

# Create the haar cascade path and map to cascade classifier
cascPath = "../cascades/data/haarcascade_frontalface_default.xml"  # sys.argv[2]

# face_cascade =cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascPath)
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()  #face recognizer module
recognizer.read("./recognizers/face-trainer.yml")  #face trained file

# create label dictionary from pickle labels
labels = {"person": 1}
with open("pickles/labels.pickle", "rb") as f:
    first_labels = pickle.load(f)
    labels = { v:k for k,v in first_labels.items()}  # we invert to use id_ as our call out value


cap = cv2.VideoCapture(0)

while True:
    # capture video frame
    ret, frame = cap.read()

    # create gray filter and map rectangle to faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    # print("Found {0} faces!".format(len(faces)))
    for (x, y, w, h) in faces:
        # print(x, y, w, h)
        roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end) (xcord_start, xcord_end)
        roi_color = frame[y:y + h, x:x + w]  # use frame or img

        img_item = 'myImg.png'
        img2_item = 'myImg2.png'
        cv2.imwrite(img_item,roi_gray)
        cv2.imwrite(img2_item,roi_color)


        # experiment with different deep learning models tensorflow/Kerra, scikit, pytorch
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            # print(id_)
            # print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 1
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, font_size, color, stroke, cv2.LINE_AA)

        color = (255, 0, 0)  # BGR 0-255
        stroke = 2  # rectangle thickness
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        eyes = eyes_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # display the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When finished or if stopped release the capture
cap.release()
cv2.destroyAllWindows()
