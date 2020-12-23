import cv2
import numpy as np


# Parameters/Functions to recall:
#   1.  cv2.imshow('windowname',frame_var_name)
#       where frame_var_name is obtained from .imread or cap.read
#       (where cap would be obtained from, e.g., cv2.VideoCapture(0))

def camera_capture():
    cap = cv2.VideoCapture(0) # param is camera number, default is 0

    while(True):
        # frame-by-frame capture; evaluates to (bool, frame)
        ret, frame = cap.read() 

        # operate on the frame (optional): NOTHING here, see below for greyscale

        # Display the resulting frame, break camera feed loop with 'q'
        cv2.imshow('frame', frame) # params: 1. var name of frame; 2. operation
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # capture release (hmmm mot sure I quite get this....)
    cap.release()
    cv2.destroyAllWindows()


def camera_capture_grey():
    cap = cv2.VideoCapture(0) # param is camera number, default is 0

    while(True):
        # frame-by-frame capture
        ret, frame = cap.read() 

        # operate on the frame (optional)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame, break camera feed loop with 'q'
        cv2.imshow('frame', gray) # params: 1. var name of frame; 2. operation
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # capture release (hmmm mot sure I quite get this....)
    cap.release()
    cv2.destroyAllWindows()


camera_capture()
