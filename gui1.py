
import cv2
import numpy as np
import os
import sys
import requests
from PIL import Image
import io


def mkGreetWindow():
	left_col = [
		[sg.Button('New User')],
		[sg.Button('Returning User')]
	]

	right_col = [
		[sg.Text("Webcam Playback (Press Space to Begin Login/Signup)")],
	    [sg.Text(size=(40, 1), key="-TOUT-")],
	    [sg.Image(key="-IMAGE-")],
	]

	layout = [[sg.Column(left_col), sg.Column(right_col)]]

	window = sg.Window('Login/Signup', layout)

	return window

def mkSignupWindow():
	left_col = [
		[sg.Text('Please Enter Your Name')],
		[sg.Input(key = '-NEWNAME-')],
		[sg.Button('Submit')]
	]

	right_col = [
		[sg.Text("Webcam Playback (Press Space to Begin Login/Signup)")],
	    [sg.Text(size=(40, 1), key="-TOUT-")],
	    [sg.Image(key="-IMAGE-")],
	]

	layout = [[sg.Column(left_col), sg.Column(right_col)]]

	window = sg.Window('Login/Signup', layout)

	return window

#### BEGIN: code from previous file #### 

def get_diff(i1, i2):
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
	J = Image.fromarray(frame) 
	binary_io = io.BtyesIO()
	J.save(binary_io, format = 'PNG') # turn the frame into binary memory resident stream (What does that mean?) 
	
	return binary_io.getvalue()

def mainlooprun():

	window = mkGreetWindow()
	cap = cv2.VideoCapture(0)
	faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	
	while True:

		event, vals = window.read()
		ret, frame = cap.read()

		if event == 'New User' or event == 'Returning User':
			faces = faceCascade.detectMultiScale(frame,1.3,5)

	        # Draw a rectangle around the face(s) in the frame
	        for (x, y, w, h) in faces:
	            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

			# repeatedly update the 'Image' in the GUI with the captured frame
			window.FindElement('-IMAGE-').Update(data = get_bytes(frame))
			# window.['-IMAGE-'].update(data = get_bytes(frame))

			if event == 'New User':
				window = mkGreetWindow()
				event1, vals1 = window.read()
				faces = faceCascade.detectMultiScale(frame,1.3,5)

		        # Draw a rectangle around the face(s) in the frame
		        for (x, y, w, h) in faces:
		            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

		        window.FindElement('-IMAGE-').Update(data = get_bytes(frame))

				if event1 == 'Submit':
					name = vals1['-NEWNAME-']




