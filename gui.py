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
        sg.InputText(),
        sg.Text("Total number of lines: "),
        sg.InputText()
    ],
    [sg.Button("Read")]
]

layout = [
    [
        file_list_column,
    ]
]

window = sg.Window(title="Image OCR", layout=layout, margins=(100,50))

while True:
    event, values = window.read()
    if event == "Read":
        print(values)
        print(values["-FILE-"])
        read.start_read(values["-FILE-"], values[0], values[1])
        #Read Here
    elif event == sg.WIN_CLOSED:
        break