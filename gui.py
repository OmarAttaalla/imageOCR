import PySimpleGUI as sg
import os
import read

font = ('Oswald', 12, 'bold')
sg.theme('DarkBlue13')
sg.set_options(font=font)

file_list_column = [
    [
        sg.Text("Select an Image:"),
        sg.In(size=(25,1), enable_events=True, key="-FILE-"),
        sg.FileBrowse(),
    ],
    [
        sg.Text("Characters per line: "),
        sg.InputText(size = (4,2)),
        sg.Text("Total number of lines: "),
        sg.InputText(size = (4,2)),
        sg.Checkbox("Dense Read", key="DENSE_READ")
    ],
    [sg.Button("Read"), sg.Text("",key="Progess")]
]

layout = [
    [
        file_list_column,
    ]
]

window = sg.Window(title="Image OCR", layout=layout, margins=(100,50))
read.pass_window(window) #Pass window to NN to make updates to GUI (Reading Progress)

Reading = False #If the Program is currently reading an image

while True:
    event, values = window.read()
    if values.pop("-OPERATION DONE-", None) != None:
        Reading = False
    if event == "Read" and Reading == False:
        Reading = True
        window.perform_long_operation(lambda : read.start_read(values["-FILE-"], values[0], values[1], values["DENSE_READ"]), '-OPERATION DONE-')
    elif event == sg.WIN_CLOSED:
        break