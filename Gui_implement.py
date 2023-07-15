# Import Needed Packages

import os
from tkinter import *
from tkinter.filedialog import askopenfilename

import customtkinter
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# Initialize Gui Window
root = customtkinter.CTk()
root.geometry("710x370")
root.resizable(True, True)
root.title('IC')
root.iconbitmap(os.path.join('assets', "icon.ico"))
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")
dark_theme = True  # Helpful In Themes Switcher
cam = True  # Helpful In Camera Status Switcher
# Define a video capture object
vid = cv2.VideoCapture(0)
opencv_image = None
# Declare the width and height in variables
width, height = 200, 200
# Set the width and height
vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Theme Switcher Button Icons Initialization
img_light = PhotoImage(file=os.path.join('assets', 'light.png'))
img_dark = PhotoImage(file=os.path.join('assets', "dark.png"))
# Camera Button Icon
cam_img = customtkinter.CTkImage(light_image=Image.open(os.path.join('assets', "camera_light.png")),
                                 dark_image=Image.open(os.path.join('assets', "camera_dark.png")))


# Function To Set Dark Theme
def dark():
    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("dark-blue")
    theme.configure(image=img_dark, bg="#242424")
    select_label.configure(bg="#242424", fg='#ebebeb')
    img_capture.configure(hover_color="#242424")


# Function To Set light Theme
def light():
    customtkinter.set_appearance_mode("light")
    customtkinter.set_default_color_theme("blue")
    theme.configure(image=img_light, bg='#ebebeb', activebackground='#ebebeb')
    select_label.configure(bg='#ebebeb', fg="#242424")
    img_capture.configure(hover_color="#ebebeb")


# Theme Switcher Function
def theme_switch():
    global dark_theme

    if dark_theme:
        light()
        dark_theme = False
    else:
        dark()
        dark_theme = True


# Open Camera Session Function
def open_camera():
    global opencv_image
    # Capture the video frame by frame
    _, frame = vid.read()
    # Convert image from one color space to other
    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Capture the latest frame and transform to image
    captured_image = Image.fromarray(opencv_image)

    # Convert captured image to photo-image
    photo_image = ImageTk.PhotoImage(image=captured_image)

    # Displaying photo-image in the label
    select_label.photo_image = photo_image

    # Configure image in the label
    select_label.configure(image=photo_image)

    # Repeat the same process after every 10 seconds
    select_label.after(1, open_camera)


# Close Camera Session Function
def close_camera():
    vid.release()
    cv2.destroyAllWindows()


# Camera Switcher Function
def switch_camera():
    global cam

    if cam:
        open_camera()
        cam = False
    else:
        close_camera()
        select_label.configure(text="\n\n\n\n\n\tSelect Input Way ", image="")


# Capture Frame From Camera And Pass It In Models To Make A Prediction
def capture():
    resize = tf.image.resize(opencv_image, (256, 256))  # Resize Frame To Model Input Size
    mood_model = load_model(os.path.join('models', 'mood.h5'))
    m_prediction = mood_model.predict(np.expand_dims(resize / 255, 0))
    if m_prediction > 0.5:
        txt = 'Mood: Sad'
        data_label.configure(text=txt)
    else:
        txt = 'Mood: Happy'
        data_label.configure(text=txt)


# Select File From PC Function
def from_file():
    link = askopenfilename()  # Initiate Link With Selected File Path
    my_img = Image.open(link)  # Open Selected Image
    resized_img = my_img.resize((200, 200))  # Resize Image
    new_img = ImageTk.PhotoImage(resized_img)  # Convert Image to PhotoImage To Place It In CTK Window
    select_label.configure(image=new_img)  # Set Image As Value Of Label
    select_label.image = new_img
    img = cv2.imread(link)  # Convert Image TO CV2 Image
    resize = tf.image.resize(img, (256, 256))  # Resize Image
    mood_model = load_model(os.path.join('models', 'mood.h5'))
    m_prediction = mood_model.predict(np.expand_dims(resize / 255, 0))
    if m_prediction > 0.5:
        txt = 'Mood: Sad'
        data_label.configure(text=txt)
    else:
        txt = 'Mood: Happy'
        data_label.configure(text=txt)


# All Labels And Buttons
select_label = Label(root, font=('calibri', 11, 'bold'), text="\n\n\n\n\n\tSelect Input Way ", bg="#242424",
                     fg='#aaaaaa')
select_label.place(x=40, y=20)
data_label = customtkinter.CTkLabel(root, font=('Impact', 25), text=" ", fg_color="transparent",
                                    justify='left')
data_label.place(x=420, y=120)
s_file = customtkinter.CTkButton(root, text='Browse', command=from_file, border_spacing=4)
s_file.place(x=75, y=230)
camera = customtkinter.CTkButton(root, text='Camera', command=switch_camera, border_spacing=4, width=110)
camera.place(x=75, y=265)
img_capture = customtkinter.CTkButton(root, image=cam_img, text=" ", command=capture, fg_color="transparent", width=40,
                                      hover_color="#242424")
img_capture.place(x=185, y=265)
theme = Button(root, text='theme', command=theme_switch, bd=0, image=img_dark, bg="#242424")
theme.place(x=665, y=330)
# Gui Main Function Call
root.mainloop()
