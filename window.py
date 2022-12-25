import time
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk

app = Tk()

app.geometry("1372x772")

def update_image():
    canvas.itemconfig(image_container, image=img2)

canvas = Canvas(
    app,
    bg = "#0a0526",
    height = 772,
    width = 1372,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
canvas.place(x = 0, y = 0)

button = ttk.Button(app, text="Update",
command=lambda:update_image())
button.pack()

img1 = PhotoImage(file="background.png")
img2 = PhotoImage(file="background1.png")

image_container = canvas.create_image(0,0, anchor="nw", image=img1)

app.resizable(False, False)
app.mainloop()