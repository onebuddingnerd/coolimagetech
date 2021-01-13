import cv2
import sys
import requests

# creating the cascade;
# a cascade is an object that has been trained on many +/-
# images (here w/ a face and w/o) and tries to distinguish
# xml for Haar Cascade is open-source and at following link
link1 = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
xml_file = open('haarcascade_frontalface_default.xml','w+')
xml_file.write(requests.get(link1).text)
xml_file.close()

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame; gives (bool, frame) of ?success, camera image
    # bool says if you have run out of frames -- irrelevant for webcame that 
    # has infinite frames til quit
    ret, frame = video_capture.read()

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # greyscale version of frame

    # detect the face(s)
    # args: 
    #   1. the frame from the video capture
    #   2. 'scaleFactor': the extent of reduction at each image scale 
    #       (scaled images are stacked upon each other in the detection process)
    #   3. 'minNeighbors': selectivity of classification -- higher values
    #       enforce more selectivity
    faces = faceCascade.detectMultiScale(frame,1.3,5)

    # Draw a rectangle around the face(s)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()