from tkinter import *
import tkinter as tk
import tkinter.font as font
import tkinter.ttk as ttk
from tkinter import *
import os
import sys
from PIL import Image, ImageTk
import time
import tkinter

window = tk.Tk()
window.title("WELCOME")
window.geometry('1920x1800')

##

##window.configure(background ="light green")
ima = PhotoImage(file = 'C:\\Users\\dhruva adithya\\Desktop\\Datasets\\img1.png')
label = tk.Label(window, image = ima)

lb3=tk.Label(window, text="UNUSUAL EVENT DETECTION",font=("Century Gothic",34,"bold","italic"),foreground="black",bg="sky blue")
lb3.place(x=500,y=50)


lb2=tk.Label(window, text=" This project presents an algorithm which is able to detect unusual event image..  ",font=("Century Gothic",14,"bold","italic"),foreground="black",bg="sky blue")
lb2.place(x=100,y=200)

lb1=tk.Label(window, text="This system will check a image and classify as Normal and Abnormal",font=("Century Gothic",14,"bold","italic"),foreground="black",bg="sky blue")
lb1.place(x=100,y=235)

lb3=tk.Label(window, text="Alerts the user as normal and abnormal action.. ",font=("Century Gothic",14,"bold","italic"),foreground="black",bg="sky blue")
lb3.place(x=100,y=270)



def fun1():
    os.system("python frnt.py")

 



btn=tk.Button(window, text="Click Here",command=fun1,width=14,height=1,font=("Century Gothic",24,"bold","italic"),foreground="black",bg="skyblue")
btn.place(x=1000,y=500)



label.pack()
window.mainloop()
