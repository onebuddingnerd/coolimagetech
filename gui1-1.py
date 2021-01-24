import cv2
import numpy as np
import os
import sys
import requests
from PIL import Image
import face_recognition
import io
import PySimpleGUI as sg
import serial


commPort = '/dev/cu.usbmodem14101'
ser = serial.Serial(commPort, baudrate = 9600, timeout = 1)
sg.theme('DarkBlue1')
width = 700
height = 600
    
def mkGreetLayout():
    left_col = [
        [sg.Button('New User')],
        [sg.Button('Returning User')],
        [sg.Button('Exit')]
    ]

    right_col = [
        [sg.Text("GROCERY GUESSER", font = "Helvetica", text_color = "goldenrod1")],
        [sg.Text("This application uses facial recognition and machine learning to guess what groceries you want!", text_color = "sandy brown")],
        [sg.Text(size=(40, 1), key="-TOUT_GREET-")],
        [sg.Image(key="-IMAGE_GREET-")]
    ]

    layout = [[sg.Column(left_col), sg.Column(right_col)]]

    return layout

def mkSignupLayout():
    left_col = [
        [sg.Text('Sign Up', key = "SignUp_Login")],
        [sg.Text('Please Enter Your Name', key = "Enter_Name", size = (30,1))],
        [sg.Input(key = '-NEWNAME-')],
        [sg.Button('Lock Face')],
        [sg.Button('Submit')]
    ]

    right_col = [
        [sg.Text(size=(40, 1), key="-TOUT-SIGNUP-")],
        [sg.Text(size = (50, 1), key = 'New_User_Registered', text_color = "yellow green")],
        [sg.Image(key="-STORED_FACE-")],
        [sg.Text('', key = 'Login_User_Verif', size = (25, 1), text_color = "yellow green")]
        #[sg.Image(key="-IMAGE_SIGNUP-")]
    ]

    layout = [[sg.Column(left_col), sg.Column(right_col)]]

    return layout

LAYOUTS = [[sg.Column(mkGreetLayout(), key = '-GREET-')],
            [sg.Column(mkSignupLayout(), key = '-SIGNUP-', visible = False)]]

#### BEGIN: code from previous file #### 

def get_diff(i1, i2): 

    face_embeddings_i2 = np.array(face_recognition.face_encodings(i2))
    face_embeddings_i1 = np.array(face_recognition.face_encodings(i1))

    return sum(sum(abs(face_embeddings_i2 - face_embeddings_i1)))

def get_min_idx(S):
        min_idx = 0
        for i in range(len(S)):
            if S[i] < S[min_idx]: min_idx = i

        return min_idx

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
                # if mode == 'debug': print('comparing with ' + name)
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

def resize_image_signup(frame):
    framePIL = cv2pil(frame)
    framePIL = framePIL.resize((200,150))
    return pil2cv(framePIL)

def resize_image_home_page(frame):
    framePIL = cv2pil(frame)
    framePIL = framePIL.resize((300,200))
    return pil2cv(framePIL)

def turnOnLED():
    ser.write(b'o')

def turnOffLED(): 
    ser.write(b'x')
    
def mainlooprun():
    window = sg.Window('Grocery Guesser',LAYOUTS, size=(width,height))
    # print('vid capture about to begin PLEASE')
    cap = cv2.VideoCapture(0)
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    iterations = 0

    playback_requested = False
    ret, frame = None, None

    freeze_frame_signup = None
    signup_frz_req = False
    x1,y1,w1,h1 = None,None,None,None

    freeze_frame_login = None
    login_frz_req = False
    x2,y2,w2,h2 = None,None,None,None
    newUser = False
    while True:
        event, vals = window.read(timeout = 20)
        ret, frame = cap.read()

        faces = faceCascade.detectMultiScale(frame, 1.3, 5)
        # Draw a rectangle around the face(s) in the frame
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)

        # repeatedly update the 'Image' in the GUI with the captured frame
        #frame_signup = frame if not signup_frz_req else freeze_frame_signup
        window.FindElement('-IMAGE_GREET-').Update(data = get_bytes(resize_image_home_page(frame)))
        if signup_frz_req:
            window.FindElement('-STORED_FACE-').Update(data = get_bytes(resize_image_signup(freeze_frame_signup)))
        if login_frz_req:
            window.FindElement('-STORED_LOGIN_FACE-').Update(data = get_bytes(resize_image_signup(freeze_frame_login)))
        #frame_login = frame if not login_frz_req else freeze_frame_login
        #window['-IMAGE_LOGIN-'].Update(data = get_bytes(resize_image(frame_login)))

        if event == 'New User':
            playback_requested = True
            # make greet window invisible and signup window visible instead
            #window['-LOGIN-'].update(visible = False)
            window['-SIGNUP-'].update(visible = True)
            window['SignUp_Login'].Update("Sign Up")
            #window['-IMAGE_GREET-'].update(data = get_bytes(resize_image_home_page(frame)))
            newUser = True
            
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
        if event == 'Submit' and newUser:
            name = vals['-NEWNAME-']
            window['New_User_Registered'].Update("New user '" + name + "' registered with this photo!")
            uncropped, cropped = freeze_frame_signup, freeze_frame_signup[y1:y1+h1,x1:x1+w1]
            cv2.imwrite('./saved_faces/'+name+'.png', uncropped)
            cv2.imwrite('./saved_faces/'+name+'_pp.png', cropped)
            newUser = False
            
        elif event == 'Submit' and not newUser:
            uncropped, cropped = freeze_frame_signup, freeze_frame_signup[y1:y1+h1,x1:x1+w1]
            diff_scores, diff_names = compute_diff_scores(uncropped, cropped)

            min_idx = get_min_idx(diff_scores)            
            name_match = diff_names[min_idx]
            if diff_scores[min_idx] < 4:
                window['Login_User_Verif'].Update('Welcome ' + name_match)
                window['New_User_Registered'].Update("")
                turnOnLED()
                
        if event == 'Returning User':
            window['-SIGNUP-'].update(visible = True)
            #window['-LOGIN-'].update(visible = True)
            window['SignUp_Login'].Update("Login")
            window['Enter_Name'].Update("Please enter your name for verification")
            window['New_User_Registered'].Update("")

        if event == 'Lock':
            if len(faces) > 0:
                bds = faces[0]
                x2,y2,w2,h2 = bds
                freeze_frame_login = frame
                login_frz_req = True

       

        if event == sg.WIN_CLOSED or event == 'Exit':
            turnOffLED()
            break

        window.refresh()

        iterations += 1
    window.close()

mainlooprun()
