
import PySimpleGUI as sg

def mkLayout1():
    left_col = [
        [sg.Button('New User')],
        [sg.Button('Returning User')],
        [sg.Button('Exit')]
    ]

    right_col = [
        [sg.Text("Webcam Playback (Press Space to Begin Login/Signup)")],
        [sg.Text(size=(40, 1), key="-TOUT-")],
        [sg.Image(key="-IMAGE-")],
    ]

    layout = [[sg.Column(left_col), sg.Column(right_col)]]

    return layout

window = sg.Window("Win1",mkLayout1())

while True:

    event, vals = window.read()

    if event == 'Exit':
        break

window.close()
