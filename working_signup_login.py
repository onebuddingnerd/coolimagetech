import cv2
import numpy as np
import os
import sys
import requests
from PIL import Image
import face_recognition

mode = 'debug' if (len(sys.argv) > 1 and sys.argv[1] == 'd') else '' 

if len(sys.argv) > 2: 
    link1 = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    xml_file = open('haarcascade_frontalface_default.xml','w+')
    xml_file.write(requests.get(link1).text)
    xml_file.close()

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

def get_diff(i1, i2): 

    face_embeddings_i2 = np.array(face_recognition.face_encodings(i2))
    face_embeddings_i1 = np.array(face_recognition.face_encodings(i1))

    return sum(sum(abs(face_embeddings_i2 - face_embeddings_i1)))

def compute_diff_scores(i1, i2): # params: full frame (a_face), crop (a_face_only)
    scores = []
    filenames = []

    users = os.listdir('./saved_faces')

    k = 0
    while (k < len(users)):
        file = users[k]
        if (file.endswith('.png')):
            if (file.endswith('_pp.png')):
                name = file[:file.index('_pp.png')]
                if mode == 'debug': print('comparing with ' + name)
                filenames.append(name)
                i3 = cv2.imread('./saved_faces/'+name+'_pp.png')
                diff = get_diff(i3, i2) # ATTENTION: passing in the full frame now
                scores.append(diff)
        k = k + 1

    return scores, filenames

def face_box_profile_save():
    def get_min_idx(S):
        min_idx = 0
        for i in range(len(S)):
            if S[i] < S[min_idx]: min_idx = i

        return min_idx

    while True:
        # Capture frame-by-frame; gives (bool, frame) of ?success, camera image
        # bool says if you have run out of frames -- irrelevant for webcame that 
        # has infinite frames til quit
        ret, frame = video_capture.read()

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

        key = cv2.waitKey(1)

        if key == ord('q'):
            break

        elif key == 32: # space key pressed: register or reauth user
            if len(faces) > 0: # a face is detected; save it unless it's already there
                a_face_bounds = faces[0] # take the first (should be only) face
                (x,y,w,h) = a_face_bounds
                a_face = frame
                a_face_only = frame[y:y+h,x:x+w] # NOTICE y-axis first
                diff_scores, diff_names = compute_diff_scores(a_face, a_face_only)

                if mode == 'debug':
                    print('comparing w', diff_names)
                    print('scores are', diff_scores)

                # deciding whether to authenticate or register
                if (len(diff_scores) == 0): # save the image(s)
                    newname = input("enter name: ")
                    cv2.imwrite('./saved_faces/'+newname+'.png', a_face)
                    cv2.imwrite('./saved_faces/'+newname+'_pp.png', a_face_only)
                    print('new user registered\n')

                else:
                    min_idx = get_min_idx(diff_scores)
                    # ask if the person is the same one as in the min-different photo
                    name_potential_match = diff_names[min_idx]
                    answer = input("Are you " + name_potential_match + "(y/n): ")
                    if answer == 'n': #save the image
                        newname = input("enter name: ")
                        cv2.imwrite('./saved_faces/'+newname+'.png', a_face)
                        cv2.imwrite('./saved_faces/'+newname+'_pp.png', a_face_only)
                        print('new user registered\n')
                    else:
                        print('welcome ' + name_potential_match)

                break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

face_box_profile_save()