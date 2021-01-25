import cv2
import numpy as np
import os
import sys
import requests
from PIL import Image
import face_recognition
import io
import PySimpleGUI as sg
import pickle
from memorizer import UserData

sg.theme('DarkBlue1')
width = 700
height = 600

all_userdata_file = './all_userdata.pickle'
ALL_USERDATA = {}
ALL_USERDATA = None
if os.path.exists(all_userdata_file) and os.path.getsize(all_userdata_file) > 0:
    ALL_USERDATA = pickle.load(open(all_userdata_file, 'rb'))
else:
    ALL_USERDATA = {}

def pickle_save(fname, data):
    with open(fname + '.pickle', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

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
        [sg.Text('', key = 'Login_User_Verif', size = (25, 1), text_color = "yellow green")],
        [sg.Column([[sg.Button('Proceed to Grocery Selection Menu')]], key = 'go-to-listmenu', visible = False)]
        # [sg.Button('')]
        # [sg.Image(key="-IMAGE_SIGNUP-")]
    ]

    layout = [[sg.Column(left_col), sg.Column(right_col)]]

    return layout

def mkGrocerySelectorLayout():
    left_col = [
        [sg.Text("Enter an item:", key = 'grocery_starttext')],
        [sg.Input(key = '-NEW_ITEM-')],
        [sg.Button('Add Item to List')],
        [sg.Button('Finished with List')],
        [sg.Button('Recommendations')],
        [sg.Button('Exit', key = 'exit2')]
    ]

    right_col = [
        [sg.Image(key = '-APPROVAL-')] # use hand cascade here
    ]

    layout = [[sg.Column(left_col), sg.Column(right_col)]]

    return layout


LAYOUTS = [[sg.Column(mkGreetLayout(), key = '-GREET-'),
            sg.Column(mkSignupLayout(), key = '-SIGNUP-', visible = False),
            sg.Column(mkGrocerySelectorLayout(), key = '-GROCERY-', visible = False)]]

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
        if file.endswith('.png'):
            if file.endswith('_pp.png'):
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
    
def mainlooprun():
    window = sg.Window('Grocery Guesser', LAYOUTS, size=(width,height))
    # print('vid capture about to begin PLEASE')
    cap = cv2.VideoCapture(0)
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    handCascade = cv2.CascadeClassifier('haar_hand.xml')
    #iterations = 0

    # Transition to grocery menu
    account_active = False #True if logged in or signed up
    proceeded = False
    currusr_data = None

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

        blur = cv2.GaussianBlur(frame,(5,5),0) 
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        retval2,thresh1 = cv2.threshold(gray,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 
        hands = handCascade.detectMultiScale(thresh1, 1.3, 6)
        mask = np.zeros(thresh1.shape, dtype = "uint8")
        
        faces = faceCascade.detectMultiScale(frame, 1.3, 5)
        # Draw a rectangle around the face(s) in the frame
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)

        # Draw a rectangle around the hand(s) in the frame
        for (x,y,w,h) in hands:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (122,122,0),2)
            cv2.rectangle(mask, (x,y),(x+w,y+h),255,-1)

        img2 = cv2.bitwise_and(thresh1, mask)
        final = cv2.GaussianBlur(img2,(7,7),0)  
        contours, hierarchy = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.drawContours(frame, contours, 0, (255,255,0), 3)
        cv2.drawContours(final, contours, 0, (255,255,0), 3)
        
        # repeatedly update the 'Image' in the GUI with the captured frame
        #frame_signup = frame if not signup_frz_req else freeze_frame_signup
        window.FindElement('-IMAGE_GREET-').Update(data = get_bytes(resize_image_home_page(frame)))
        window.FindElement('-STORED_FACE-').Update(data = get_bytes(resize_image_signup(frame)))
        if signup_frz_req:
            window.FindElement('-STORED_FACE-').Update(data = get_bytes(resize_image_signup(freeze_frame_signup)))
        if login_frz_req:
            window.FindElement('-STORED_LOGIN_FACE-').Update(data = get_bytes(resize_image_signup(freeze_frame_login)))
        #frame_login = frame if not login_frz_req else freeze_frame_login
        #window['-IMAGE_LOGIN-'].Update(data = get_bytes(resize_image(frame_login)))

        ### BEGIN: CODE FOR TRANSITION TO GROCERY MENU ###

        if account_active:
            window['go-to-listmenu'].update(visible = True)

        if event == 'Proceed to Grocery Selection Menu':
        # a login or signup has happened and then the proceed button was pressed
            window['-GREET-'].update(visible = False)
            window['-SIGNUP-'].update(visible = False)
            window['-GROCERY-'].update(visible = True)

        ### END: CODE FOR TRANSITION TO GROCERY MENU ###

        ### BEGIN: CODE FOR GROCERY MENU ###

        if event == 'Add Item to List':
            new_item = vals['-NEW_ITEM-']
            currusr_data.add_product(new_item)
            currusr_data.debug_print()

        if event == 'Finished with List':
            currusr_data.list_reset()

        if event == 'Recommendations':
            print(currusr_data.get_ordered_recs())

        ### END: CODE FOR GROCERY MENU ###

        if event == 'New User':
            playback_requested = True
            # make greet window invisible and signup window visible instead
            window['-GREET-'].update(visible = False)
            window['-SIGNUP-'].update(visible = True)
            window['SignUp_Login'].Update("Sign Up")
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
            account_active = True
            uncropped, cropped = freeze_frame_signup, freeze_frame_signup[y1:y1+h1,x1:x1+w1]
            cv2.imwrite('./saved_faces/'+name+'.png', uncropped)
            cv2.imwrite('./saved_faces/'+name+'_pp.png', cropped)
            ALL_USERDATA[name] = UserData(name)
            currusr_data = ALL_USERDATA[name]
            newUser = False
            
        elif event == 'Submit' and not newUser:
            uncropped, cropped = freeze_frame_signup, freeze_frame_signup[y1:y1+h1,x1:x1+w1]
            diff_scores, diff_names = compute_diff_scores(uncropped, cropped)

            min_idx = get_min_idx(diff_scores)            
            name_match = diff_names[min_idx]
            if diff_scores[min_idx] < 4:
                window['Login_User_Verif'].Update('Welcome ' + name_match)
                window['New_User_Registered'].Update("")
                currusr_data = ALL_USERDATA[name_match]
                account_active = True

        if event == 'Returning User':
            window['-GREET-'].update(visible = False)
            window['-SIGNUP-'].update(visible = True)
            window['SignUp_Login'].Update("Login")
            window['Enter_Name'].Update("Please enter your name for verification")
            window['New_User_Registered'].Update("")

        if event == 'Lock':
            if len(faces) > 0:
                bds = faces[0]
                x2,y2,w2,h2 = bds
                freeze_frame_login = frame
                login_frz_req = True

        if event == sg.WIN_CLOSED or event == 'Exit' or event == 'exit2':
            break

        window.refresh()

        #iterations += 1

    pickle_save('all_userdata', ALL_USERDATA)
    window.close()

mainlooprun()
