import numpy as np 
import cv2

cap = cv2.VideoCapture(0)

while(True):
    #capture video frame
    ret, frame = cap.read()

    # create gray filter and map rectangle to faces
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    print("Found {0} faces!".format(len(faces)))
    for (x, y, w, h) in faces:
    	#print(x,y,w,h)
    	roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end) (xcord_start, xcord_end)
    	roi_color = frame[y:y+h, x:x+w]

    #display the frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When finished or if false release the capture
cap.release()
cv2.destroyAllWindows()