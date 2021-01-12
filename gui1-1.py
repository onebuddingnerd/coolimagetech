import cv2
import numpy as np
import os
import sys
import requests
from PIL import Image
import face_recognition
import io
import PySimpleGUI as sg
#Hi 

def mkGreetLayout():
    left_col = [
        [sg.Button('New User')],
        [sg.Button('Returning User')]
    ]

    right_col = [
        [sg.Text("Webcam Playback (Press Space to Begin Login/Signup)")],
        [sg.Text(size=(40, 1), key="-TOUT_GREET-")],
        [sg.Image(key="-IMAGE_GREET-")],
    ]

    layout = [[sg.Column(left_col), sg.Column(right_col)]]

    return layout

def mkSignupLayout():
    left_col = [
        [sg.Text('Please Enter Your Name')],
        [sg.Input(key = '-NEWNAME-')],
        [sg.Button('Submit')],
        [sg.Button('Lock Face')],
        [sg.Button('Exit1')]
    ]

    right_col = [
        [sg.Text("Webcam Playback (Press Space to Begin Login/Signup)")],
        [sg.Text(size=(40, 1), key="-TOUT-SIGNUP-")],
        [sg.Image(key="-IMAGE_SIGNUP-")]
    ]

    layout = [[sg.Column(left_col), sg.Column(right_col)]]

    return layout

def mkLoginLayout():
    # DO LATER
    return [[]]

LAYOUTS = [[sg.Column(mkGreetLayout(), key = '-GREET-')],
            [sg.Column(mkSignupLayout(), key = '-SIGNUP-', visible = False)],
            [sg.Column(mkLoginLayout(), key = '-LOGIN-', visible = False)]]

#### BEGIN: code from previous file #### 

def get_diff(i1, i2): 

    face_embeddings_i2 = np.array(face_recognition.face_encodings(i2))
    face_embeddings_i1 = np.array(face_recognition.face_encodings(i1))

    return sum(sum(abs(face_embeddings_i2 - face_embeddings_i1)))


def compute_diff_scores(i1, i2): # params: full frame (a_face), crop(a_face_only)
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
                diff = get_diff(i3, i2)
                scores.append(diff)
        k = k + 1

    return scores, filenames

#### END: code from previous file #### 

# takes frame and returns bytes of frame compatible with cv
def get_bytes(frame):
    return cv2.imencode('.png', frame)[1].tobytes()

def cv2pil(cv):
        colorconv_cv = cv2.cvtColor(cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(colorconv_cv)

def pil2cv(pil):
    cv2im = np.array(pil)
    return cv2im[:,:,::-1] # reversing the z-axis (color channels: RGB -> BGR)

def resize_image(frame):
    framePIL = cv2pil(frame)
    framePIL = framePIL.resize((600,400))
    return pil2cv(framePIL)


def mainlooprun():

    window = sg.Window('Login App',LAYOUTS)
    # print('vid capture about to begin PLEASE')
    cap = cv2.VideoCapture(0)
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    iterations = 0

    playback_requested = False
    ret, frame = None, None
    freeze_frame_signup = None
    signup_frz_req = False
    x1,y1,w1,h1 = None,None,None,None
    
    while True:
        event, vals = window.read(timeout = 20)
        # ret, frame = cap.read()
        # print("event:", event, iterations)
        # ret, frame = cap.read()
        

        if playback_requested:
            ret, frame = cap.read()

            faces = faceCascade.detectMultiScale(frame, 1.3, 5)
            # print(len(faces))

            # Draw a rectangle around the face(s) in the frame
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)

            # repeatedly update the 'Image' in the GUI with the captured frame
            frame1 = frame if not signup_frz_req else freeze_frame_signup
            window.FindElement('-IMAGE_SIGNUP-').Update(data = get_bytes(resize_image(frame1)))
            window.FindElement('-IMAGE_GREET-').Update(data = get_bytes(resize_image(frame)))


        if event == 'New User':

            playback_requested = True

            # make greet window invisible and signup window visible instead
            window['-GREET-'].update(visible = False)
            window['-SIGNUP-'].update(visible = True)

            # window.['-IMAGE-'].update(data = get_bytes(frame))

        # window.FindElement('-IMAGE_SIGNUP-').Update(data = get_bytes(resize_image(frame)))
        # print(faces)

        if event == 'Lock Face':
            if len(faces) > 0:
                bds = faces[0]
                x1,y1,w1,h1 = bds
                # add code to stop updating the frame
                freeze_frame_signup = frame
                signup_frz_req = True

        #newname = input("enter name: ")
        if event == 'Submit':
            name = vals['-NEWNAME-']
            print(name)
            # x,y,w,h = faces[0]
            uncropped, cropped = freeze_frame_signup, freeze_frame_signup[y1:y1+h1,x1:x1+w1]
            cv2.imwrite('./saved_faces/'+name+'.png', uncropped)
            cv2.imwrite('./saved_faces/'+name+'_pp.png', cropped)
            print('new user registered\n')
            

        if event == 'Returning User': # Returning User
            window['-GREET-'].update(visible = False)
            window['-LOGIN-'].update(visible = True)

            if len(faces) == 0:
                bds = faces[0]
                x,y,w,h = bds
                uncropped, cropped = frame, frame[y:y+h,x:x+w]
                diff_scores, diff_names = compute_diff_scores(uncropped, cropped)

                min_idx = get_min_idx(diff_scores)
                # ask if the person is the same one as in the min-different photo
                name_match = diff_names[min_idx]

                print('Welcome ' + name_match)
                break

        if event == sg.WIN_CLOSED or event == 'Exit1':
            break

        window.refresh()

        iterations += 1
    window.close()

mainlooprun()
