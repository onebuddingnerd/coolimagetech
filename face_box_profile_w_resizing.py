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


def get_diff1(i1, i2):
    def cv2pil(cv):
        colorconv_cv = cv2.cvtColor(cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(colorconv_cv)

    def pil2cv(pil):
        cv2im = np.array(pil)
        return cv2im[:,:,::-1] # reversing the z-axis (color channels: RGB -> BGR)

    # resize the two images (smaller one unchanged, larger one shrunken)
    ymax, xmax = min(i2.shape[0], i1.shape[0]), min(i2.shape[1], i1.shape[1])

    i1_pil_resized = (cv2pil(i1)).resize((xmax,ymax)) 
    i1_resized_cv = pil2cv(i1_pil_resized)

    i2_pil_resized = (cv2pil(i2)).resize((xmax,ymax))
    i2_resized_cv = pil2cv(i2_pil_resized)

    # use mean value across channels
    i1r, i1g, i1b = i1_resized_cv[:,:,0], i1_resized_cv[:,:,1], i1_resized_cv[:,:,2]
    i1_1 = np.array([[(i1r[i,j] + i1g[i,j] + i1b[i,j])/3  for j in range (i1_resized_cv.shape[1])] for i in range(i1_resized_cv.shape[0])], dtype = 'int64')
    i2r, i2g, i2b = i2_resized_cv[:,:,0], i2_resized_cv[:,:,1], i2_resized_cv[:,:,2]
    i2_1 = np.array([[(i2r[i,j] + i2g[i,j] + i2b[i,j])/3  for j in range (i2_resized_cv.shape[1])] for i in range(i2_resized_cv.shape[0])], dtype = 'int64')

    # The normalized Sum of Square Difference -- This is pretty bad in the cases I tested
    # sq_diff = (i2_1-i1_1)**2
    # sum_sq_dff = sum(sum(sq_diff))
    # normalize = ((sum(sum(i1_1**2))) + (sum(sum(i1_1**2))))**(.5)

    # The difference in the L2 Norms
    L2_i1 = (sum(sum(i1_1**2)))**(0.5)
    L2_i2 = (sum(sum(i2_1**2)))**(0.5)

    return abs(L2_i2 - L2_i1)


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