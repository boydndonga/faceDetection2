import numpy as np 
import cv2

cap = cv2.VideoCapture(0)

while(True):
    #capture video frame
    ret, frame = cap.read()

    #display the frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When finished or if false release the capture
cap.release()
cv2.destroyAllWindows()