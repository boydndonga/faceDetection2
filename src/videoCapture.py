import numpy as np 
import cv2


# Create the haar cascade path and map to cascade classifier
cascPath = "../cascades/data/haarcascade_frontalface_default.xml" #sys.argv[2]
# face_cascade =cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascPath)
cap = cv2.VideoCapture(0)

while(True):
    #capture video frame
    ret, frame = cap.read()

    # create gray filter and map rectangle to faces
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    # print("Found {0} faces!".format(len(faces)))
    for (x, y, w, h) in faces:
    	print(x,y,w,h)
    	roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end) (xcord_start, xcord_end)
        # img_item = 'myImg.png'
        # cv2.imwrite(img_item,roi_gray)
    	roi_color = frame[y:y+h, x:x+w]
        color = (255, 0, 0) #BGR 0-255 
    	stroke = 2 #rectangle thickness
    	end_cord_x = x + w
    	end_cord_y = y + h
    	cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    #display the frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When finished or if false release the capture
cap.release()
cv2.destroyAllWindows()