import tkinter as tk
from tkinter import *
from tkinter import filedialog
import cv2
import json
import os
from PIL import Image, ImageTk
from attendance import recognition, markAttendanceInSheet
from training import train
from faceDetection import extractFaces

### Global Variables
uploadedImageFilePath = ""
# Create a placeholder for the image object
photo = None
newMemberDirectory = None
newMemberName = None
newMemberRoll = None

# Create the main window
root = tk.Tk()

root.title("Attendance System Application")

title = Label(root, text = "Face Recognition based Attendance System", font = ("Arial", 26), anchor = "center")
# title.place(x = 0, y = 0)
title.pack()

# Load the image file
bg_image = tk.PhotoImage(file="utils/background.png")

# Create a canvas widget and place it in the main window
canvas = tk.Canvas(root, width=900, height=600)
canvas.pack()

input_image=Label(canvas,text="Input Image",font = ("Arial", 15), anchor = "center")
input_image.place(x=175,y=10)


results=Label(canvas,text="Results",font = ("Arial", 15), anchor = "center")
results.place(x=610,y=10)


#Set the resizable property False
root.resizable(False, False)

# Place the image on the canvas
canvas.create_image(0, 0, image=bg_image, anchor='nw')




### Image canvas
newCanvas = tk.Canvas(canvas, width = 400, height = 400)
newCanvas.place(x = 40, y = 40)




#### --------------- Result Window -----------------------------

# Create a Canvas widget with a Scrollbar
resultCanvas = Canvas(canvas, width=400, height=400)
scrollbar = Scrollbar(canvas, command=resultCanvas.yview)
resultCanvas.place(x = 450, y = 40)
scrollbar.place(x = 850, y = 40, height = 400)
resultCanvas.configure(yscrollcommand=scrollbar.set)

############################


# Define a function to load and display an image
def load_image():
    global photo
    global uploadedImageFilePath

    # Open the image file
    uploadedImageFilePath = tk.filedialog.askopenfilename()
    

    uploadedImageName = uploadedImageFilePath

    image = Image.open(uploadedImageFilePath)

    # Resize the image to fit in the window
    width, height = image.size
    if width > 400 or height > 400:
        ratio = min(400/width, 400/height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        image = image.resize((new_width, new_height))

    # Convert the image to a PhotoImage object
    photo = ImageTk.PhotoImage(image)

    # Delete any existing image on the canvas
    newCanvas.delete("all")

    # Place the image on the newCanvas
    newCanvas.create_image(0, 0, image=photo, anchor='nw')



# function to handle the detection calls and gets the data back
def handleAttendance():
    global uploadedImageFilePath
    results = recognition(uploadedImageFilePath)
    # results = [{"Name":"Kaushal", "Roll":224101031}]
    resultCanvas.delete("all")


    # Create a Frame on the Canvas to hold the dynamic labels
    frame = Frame(resultCanvas)
    resultCanvas.create_window((0, 0), window=frame, anchor='nw')
    
    headers = ["Name", "Roll Number"]

    for j, header in enumerate(headers):
        label = Label(frame, text=header, borderwidth=1, relief="solid", font=("Arial", 11, "bold"))
        label.grid(row=0, column=j, sticky="nsew")

    # Create some dynamic labels
    for i, stud in enumerate(results):
        label = Label(frame, text=stud['Name'], borderwidth=1, relief="solid")
        label.grid(row=i+1, column=0, sticky="nsew")

        label = Label(frame, text=stud['Roll'], borderwidth=1, relief="solid")
        label.grid(row=i+1, column=1, sticky="nsew")

    # Update the scroll region of the Canvas
    frame.update_idletasks()
    frame.grid_columnconfigure(0, minsize=200)
    frame.grid_columnconfigure(1, minsize=200)
    resultCanvas.config(scrollregion=resultCanvas.bbox("all"))

    markAttendanceInSheet(results)


def browse_folder(entry):
    # Open a folder dialog and set the value of the entry to the selected folder
    foldername = filedialog.askdirectory()
    entry.configure(state="normal")
    entry.delete(0, END)
    entry.insert(0, foldername)
    entry.configure(state="readonly")


def open_popup():
    # Create a new Toplevel window
    popup = Toplevel()

    # Set the title of the popup window
    popup.title("Enrollment Form")

    popup.geometry("500x200")

    # Add form elements to the popup window
    name_label = Label(popup, text="Name: ")
    # name_label.grid(row=0, column=0)
    name_label.place(x = 10, y = 10)
    name_entry = Entry(popup)
    # name_entry.grid(row=0, column=1)
    name_entry.place(x = 150, y = 10)

    email_label = Label(popup, text="Roll No: ")
    # email_label.grid(row=2, column=0)
    email_label.place(x = 10, y = 50)
    email_entry = Entry(popup)
    # email_entry.grid(row=2, column=1)
    email_entry.place(x = 150, y = 50)

    # path_label = Label(popup, text="Images Path: ")
    # path_label.grid(row=4, column=0)
    # path_entry = Entry(popup)
    # path_entry.grid(row=4, column=1)

    # Add a file selection entry to the popup window
    path_label = Label(popup, text="File:")
    # path_label.grid(row=4, column=0)
    path_label.place(x = 10, y = 100)
    path_entry = Entry(popup, state="readonly")
    # path_entry.grid(row=4, column=1)
    path_entry.place(x = 150, y = 100)

    path_button = Button(popup, text="Browse...", command=lambda: browse_folder(path_entry))
    # path_button.grid(row=4, column=2)
    path_button.place(x = 350, y = 100)

    def get_data_and_close():
        global newMemberName
        global newMemberRoll
        global newMemberDirectory
        newMemberName = name_entry.get()
        newMemberRoll = email_entry.get()
        newMemberDirectory = path_entry.get()
        popup.destroy()
        
        updateDataSet(newMemberDirectory, newMemberName)
        extractFaces()
        train()

    # Add a submit button to the popup window
    # submit_button = Button(popup, text="Submit")
    submit_button = Button(popup, text="Submit", command = get_data_and_close)
    # submit_button.grid(row=6, column=1)
    submit_button.place(x = 150, y = 150)

    

def updateDataSet(directory, name):
    path = os.path.join("dataset", name)
    if not os.path.exists(path):
        os.mkdir(path)
    
    # print(path)
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            # print(os.path.join(dirname, filename))
            out_path = os.path.join(path, filename)
            img = cv2.imread(os.path.join(dirname, filename))
            if img is None:
                continue

            cv2.imwrite(out_path, img)




############## ------------------ Buttons ------------------
# Create a button to load an imageutton(root, text="Load Image", command=load_image)
loadImage = PhotoImage(file="utils/icons8-add-image-48.png")
loadImageButton = tk.Button(canvas, text="Load Image", command=load_image, compound="top", image=loadImage)
loadImageButton.place(x = 40, y = 470)


attendanceImage = PhotoImage(file="utils/icons8-done-50.png")
markAttendanceButton = tk.Button(canvas, text="Mark Attendance", command = handleAttendance, compound="top", image=attendanceImage)
markAttendanceButton.place(x = 200, y = 470)


enrollImage = PhotoImage(file="utils/icons8-enrollment-64.png")
addStudentButton = Button(canvas, text="Enrollment", compound="top", image=enrollImage, command = open_popup)
addStudentButton.place(x = 763, y = 470)



# Run the main event loop
root.mainloop()
