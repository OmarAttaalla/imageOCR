import PySimpleGUI as sg
import os
import read

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
        sg.InputText(size = (4,2))
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

while True:
    event, values = window.read()
    if event == "Read":
        print(values)
        print(values["-FILE-"])
        window.perform_long_operation(lambda : read.start_read(values["-FILE-"], values[0], values[1]), '-OPERATION DONE-')
    elif event == sg.WIN_CLOSED:
        break