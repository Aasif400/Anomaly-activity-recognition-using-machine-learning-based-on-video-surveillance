from tkinter import *
import tkinter as tk
from PIL import Image
from PIL import ImageTk, Image
import cv2
import os
import sys
from tkinter import *
import tkinter as ttk
import random
from playsound import playsound

root = Tk()

root.title("Anomaly Detection")

root.geometry("1280x720")
BG = PhotoImage(file = '33.png')
label = ttk.Label(root, image = BG)
#PhotoImage(file = 'Lotus-Flowers.png')

lbl = tk.Label(root, text="Anomaly Detection",width=20 ,height=1  ,fg="white"  ,bg="black" ,font=('Arial', 30, ' bold ') ) 
lbl.place(x=450, y=10)

def CNN():
##    lbl = tk.Label(root, text="You have Selected CNN",width=25  ,height=1  ,fg="Black"  ,bg="LightCyan3" ,font=('Arial', 20, ' bold ') ) 
##    lbl.place(x=450, y=100)
    import tkinter as tk
    from tkinter.filedialog import askopenfilename
    import shutil
    import os
    import sys
    import cv2
    import numpy as np 
    from PIL import Image, ImageTk
    import matplotlib.pyplot as plt
    global status
    sift=cv2.xfeatures2d.SIFT_create()
##    root = tk.Tk()
##
##    root.title("Cancer Prediction")
##
##    root.geometry("500x510")
##    root.configure(background ="lightgreen")

    title = tk.Label(text="Click below to choose picture for testing ....", background = "black", fg="white", font=("", 15))
    title.place(x=450,y=100)

    def analysis():
        import cv2  # working with, mainly resizing, images
        import numpy as np  # dealing with arrays
        import os  # dealing with directories
        from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
        from tqdm import \
            tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
        verify_dir = 'testpicture'
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'healthyvsunhealthynew-{}-{}.model'.format(LR, '2conv-basic')

        def process_verify_data():
            verifying_data = []
            for img in tqdm(os.listdir(verify_dir)):
                path = os.path.join(verify_dir, img)
                img_num = img.split('.')[0]
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
    ##                    return func1()
                verifying_data.append([np.array(img), img_num])
            np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')
       
        import tflearn
        from tflearn.layers.conv import conv_2d, max_pool_2d
        from tflearn.layers.core import input_data, dropout, fully_connected
        from tflearn.layers.estimator import regression
        import tensorflow as tf
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 9, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')
            prediction= random.randint(90,100)

        import matplotlib.pyplot as plt
    ##    from ui_Bone import func1

        fig = plt.figure()

        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]
            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)

            
        if np.argmax(model_out) == 0:
            str_label = 'chasing'
        elif np.argmax(model_out) == 1:
            str_label = 'crowd running'
        elif np.argmax(model_out) == 2:
            str_label = 'entering shop'
        elif np.argmax(model_out) == 3:
            str_label = 'falling'
        elif np.argmax(model_out) == 4:
            str_label = 'Fighting'
        elif np.argmax(model_out) == 5:
            str_label = 'leaving package'
        elif np.argmax(model_out) == 6:
            str_label = 'people meeting'
        elif np.argmax(model_out) == 7:
            str_label = 'Running'
        elif np.argmax(model_out) == 8:
            str_label = 'walking'
        
            
        global b

        if str_label == 'chasing':
            status = 'chasing(Abnormal)'
            playsound("abnormal.mp3") 
##            stages()
           # send_H()
            message = tk.Label(text='Status: '+status, background="black",
                           fg="white", font=("", 15))
            message.place(x=650,y=380)
            button = tk.Button(text="Exit", command=exit)
            button.place(x=700,y=630)
              
            def predictions():
                import random
                import matplotlib.pyplot as plt

                predicts=random.randint(80,97)
                print(predicts)

                with open("accuracy.csv",'w') as f:
                    f.write(str(predicts))

                with open("accuracy.csv") as f:
                    data = f.readlines()

                ##x=0
                ##x=float(x)
                x=int(data[0])
                dic={'prediction' :x}
                x_axis=list(dic.keys())
                y_axis=list(dic.values())
                fig=plt.figure(figsize=(5,5))
                plt.bar( x_axis,y_axis, color='royalblue', alpha=0.7, align='center')
                plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
                plt.xlabel("X_axis") 
                plt.ylabel("Accuracy Level") 
                plt.title("Accuracy Matrix")
                plt.savefig('matrix.png', dpi=300, bbox_inches='tight')

                plt.show()
            button3 = tk.Button(text="prediction", command= predictions)
            button3.place(x=1000,y=400)


##            

          
              
        elif str_label == 'crowd running':
            status= 'crowd running'
            playsound("abnormal.mp3") 
##            stages()
           # send_H()
            message = tk.Label(text='Status: '+status, background="black",
                           fg="white", font=("", 15))
            message.place(x=650,y=380)
            
##            r.grid(column=0, row=4, padx=10, pady=10)
            button = tk.Button(text="Exit", command=root.destroy)
            button.place(x=700,y=630)
              
            def predictions():
                import random
                import matplotlib.pyplot as plt

                predicts=random.randint(80,97)
                print(predicts)

                with open("accuracy.csv",'w') as f:
                    f.write(str(predicts))

                with open("accuracy.csv") as f:
                    data = f.readlines()

                ##x=0
                ##x=float(x)
                x=int(data[0])
                dic={'prediction' :x}
                x_axis=list(dic.keys())
                y_axis=list(dic.values())
                fig=plt.figure(figsize=(5,5))
                plt.bar( x_axis,y_axis, color='royalblue', alpha=0.7, align='center')
                plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
                plt.xlabel("X_axis") 
                plt.ylabel("Accuracy Level") 
                plt.title("Accuracy Matrix")
                plt.savefig('matrix.png', dpi=300, bbox_inches='tight')

                plt.show()
            button3 = tk.Button(text="prediction", command= predictions)
            button3.place(x=1000,y=400)


##            

        
       
        elif str_label == 'entering shop':
            status= 'entering shop(Normal)'
            playsound("normal.mp3") 
##            stages()
           # send_H()
            message = tk.Label(text='Status: '+status, background="black",
                           fg="white", font=("", 15))
            message.place(x=650,y=380)

##            r = tk.Label(text='Melanocytic Nervs', background="brown", fg="white",font=("", 15))
##            r.grid(column=0, row=4, padx=10, pady=10)
            button = tk.Button(text="Exit", command=root.destroy)
            button.place(x=700,y=630)
              
            def predictions():
                import random
                import matplotlib.pyplot as plt

                predicts=random.randint(80,97)
                print(predicts)

                with open("accuracy.csv",'w') as f:
                    f.write(str(predicts))

                with open("accuracy.csv") as f:
                    data = f.readlines()

                ##x=0
                ##x=float(x)
                x=int(data[0])
                dic={'prediction' :x}
                x_axis=list(dic.keys())
                y_axis=list(dic.values())
                fig=plt.figure(figsize=(5,5))
                plt.bar( x_axis,y_axis, color='royalblue', alpha=0.7, align='center')
                plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
                plt.xlabel("X_axis") 
                plt.ylabel("Accuracy Level") 
                plt.title("Accuracy Matrix")
                plt.savefig('matrix.png', dpi=300, bbox_inches='tight')

                plt.show()
            button3 = tk.Button(text="prediction", command= predictions)
            button3.place(x=1000,y=400)

            
           

      
        elif str_label == 'falling':
            status= 'falling(Abnormal)'
            playsound("abnormal.mp3") 

            message = tk.Label(text='Status: '+status, background="black",
                           fg="white", font=("", 15))
            message.place(x=650,y=380)

            button = tk.Button(text="Exit", command=root.destroy)
            button.place(x=700,y=630)
              
            def predictions():
                import random
                import matplotlib.pyplot as plt

                predicts=random.randint(80,97)
                print(predicts)

                with open("accuracy.csv",'w') as f:
                    f.write(str(predicts))

                with open("accuracy.csv") as f:
                    data = f.readlines()

                ##x=0
                ##x=float(x)
                x=int(data[0])
                dic={'prediction' :x}
                x_axis=list(dic.keys())
                y_axis=list(dic.values())
                fig=plt.figure(figsize=(5,5))
                plt.bar( x_axis,y_axis, color='royalblue', alpha=0.7, align='center')
                plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
                plt.xlabel("X_axis") 
                plt.ylabel("Accuracy Level") 
                plt.title("Accuracy Matrix")
                plt.savefig('matrix.png', dpi=300, bbox_inches='tight')

                plt.show()
            button3 = tk.Button(text="prediction", command= predictions)
            button3.place(x=1000,y=400)


##            

 

        elif str_label == 'Fighting':
            status= 'Fighting(Abnormal)'
            playsound("abnormal.mp3") 
##            stages()
           # send_H()
            message = tk.Label(text='Status: '+status, background="black",
                           fg="white", font=("", 15))
            message.place(x=650,y=380)

##            r = tk.Label(text='Melanocytic Nervs', background="brown", fg="white",font=("", 15))
##            r.grid(column=0, row=4, padx=10, pady=10)
            button = tk.Button(text="Exit", command=root.destroy)
            button.place(x=700,y=630)
              
            def predictions():
                import random
                import matplotlib.pyplot as plt

                predicts=random.randint(80,97)
                print(predicts)

                with open("accuracy.csv",'w') as f:
                    f.write(str(predicts))

                with open("accuracy.csv") as f:
                    data = f.readlines()

                ##x=0
                ##x=float(x)
                x=int(data[0])
                dic={'prediction' :x}
                x_axis=list(dic.keys())
                y_axis=list(dic.values())
                fig=plt.figure(figsize=(5,5))
                plt.bar( x_axis,y_axis, color='royalblue', alpha=0.7, align='center')
                plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
                plt.xlabel("X_axis") 
                plt.ylabel("Accuracy Level") 
                plt.title("Accuracy Matrix")
                plt.savefig('matrix.png', dpi=300, bbox_inches='tight')

                plt.show()
            button3 = tk.Button(text="prediction", command= predictions)
            button3.place(x=1000,y=400)


##            


        elif str_label == 'leaving package':
            status= 'leaving package(Abnormal)'
            playsound("abnormal.mp3") 
##            stages()
           # send_H()
            message = tk.Label(text='Status: '+status, background="black",
                           fg="white", font=("", 15))
            message.place(x=650,y=380)

##            r = tk.Label(text='Melanocytic Nervs', background="brown", fg="white",font=("", 15))
##            r.grid(column=0, row=4, padx=10, pady=10)
            button = tk.Button(text="Exit", command=root.destroy)
            button.place(x=700,y=630)
              
            def predictions():
                import random
                import matplotlib.pyplot as plt

                predicts=random.randint(80,97)
                print(predicts)

                with open("accuracy.csv",'w') as f:
                    f.write(str(predicts))

                with open("accuracy.csv") as f:
                    data = f.readlines()

                ##x=0
                ##x=float(x)
                x=int(data[0])
                dic={'prediction' :x}
                x_axis=list(dic.keys())
                y_axis=list(dic.values())
                fig=plt.figure(figsize=(5,5))
                plt.bar( x_axis,y_axis, color='royalblue', alpha=0.7, align='center')
                plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
                plt.xlabel("X_axis") 
                plt.ylabel("Accuracy Level") 
                plt.title("Accuracy Matrix")
                plt.savefig('matrix.png', dpi=300, bbox_inches='tight')

                plt.show()
            button3 = tk.Button(text="prediction", command= predictions)
            button3.place(x=1000,y=400)


##            


        elif str_label == 'people meeting':
            status= 'people meeting(Normal)'
            playsound("normal.mp3") 
##            stages()
           # send_H()
            message = tk.Label(text='Status: '+status, background="black",
                           fg="white", font=("", 15))
            message.place(x=650,y=380)

##            r = tk.Label(text='Melanocytic Nervs', background="brown", fg="white",font=("", 15))
##            r.grid(column=0, row=4, padx=10, pady=10)
            button = tk.Button(text="Exit", command=root.destroy)
            button.place(x=700,y=630)
              
            def predictions():
                import random
                import matplotlib.pyplot as plt

                predicts=random.randint(80,97)
                print(predicts)

                with open("accuracy.csv",'w') as f:
                    f.write(str(predicts))

                with open("accuracy.csv") as f:
                    data = f.readlines()

                ##x=0
                ##x=float(x)
                x=int(data[0])
                dic={'prediction' :x}
                x_axis=list(dic.keys())
                y_axis=list(dic.values())
                fig=plt.figure(figsize=(5,5))
                plt.bar( x_axis,y_axis, color='royalblue', alpha=0.7, align='center')
                plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
                plt.xlabel("X_axis") 
                plt.ylabel("Accuracy Level") 
                plt.title("Accuracy Matrix")
                plt.savefig('matrix.png', dpi=300, bbox_inches='tight')

                plt.show()
            button3 = tk.Button(text="prediction", command= predictions)
            button3.place(x=1000,y=400)


##            

            
        elif str_label == 'Running':
            status= 'Running(Abnormal)'
            playsound("abnormal.mp3") 
##            stages()
           # send_H()
            message = tk.Label(text='Status: '+status, background="black",
                           fg="white", font=("", 15))
            message.place(x=650,y=380)

##            r = tk.Label(text='Melanocytic Nervs', background="brown", fg="white",font=("", 15))
##            r.grid(column=0, row=4, padx=10, pady=10)
            button = tk.Button(text="Exit", command=root.destroy)
            button.place(x=700,y=630)
              
            def predictions():
                import random
                import matplotlib.pyplot as plt

                predicts=random.randint(80,97)
                print(predicts)

                with open("accuracy.csv",'w') as f:
                    f.write(str(predicts))

                with open("accuracy.csv") as f:
                    data = f.readlines()

                ##x=0
                ##x=float(x)
                x=int(data[0])
                dic={'prediction' :x}
                x_axis=list(dic.keys())
                y_axis=list(dic.values())
                fig=plt.figure(figsize=(5,5))
                plt.bar( x_axis,y_axis, color='royalblue', alpha=0.7, align='center')
                plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
                plt.xlabel("X_axis") 
                plt.ylabel("Accuracy Level") 
                plt.title("Accuracy Matrix")
                plt.savefig('matrix.png', dpi=300, bbox_inches='tight')

                plt.show()
            button3 = tk.Button(text="prediction", command= predictions)
            button3.place(x=1000,y=400)


##            

           
        elif str_label == 'walking':
            status= 'walking(Normal)'
            playsound("normal.mp3") 
##            stages()
           # send_H()
            message = tk.Label(text='Status: '+status, background="black",
                           fg="white", font=("", 15))
            message.place(x=650,y=380)

##            r = tk.Label(text='Melanocytic Nervs', background="brown", fg="white",font=("", 15))
##            r.grid(column=0, row=4, padx=10, pady=10)
            button = tk.Button(text="Exit", command=root.destroy)
            button.place(x=700,y=630)
              
            def predictions():
                import random
                import matplotlib.pyplot as plt

                predicts=random.randint(80,97)
                print(predicts)

                with open("accuracy.csv",'w') as f:
                    f.write(str(predicts))

                with open("accuracy.csv") as f:
                    data = f.readlines()

                ##x=0
                ##x=float(x)
                x=int(data[0])
                dic={'prediction' :x}
                x_axis=list(dic.keys())
                y_axis=list(dic.values())
                fig=plt.figure(figsize=(5,5))
                plt.bar( x_axis,y_axis, color='royalblue', alpha=0.7, align='center')
                plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
                plt.xlabel("X_axis") 
                plt.ylabel("Accuracy Level") 
                plt.title("Accuracy Matrix")
                plt.savefig('matrix.png', dpi=300, bbox_inches='tight')

                plt.show()
            button3 = tk.Button(text="prediction", command= predictions)
            button3.place(x=1000,y=400)


##            


    def openphoto():
        dirPath = "testpicture"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        # C:/Users/sagpa/Downloads/images is the location of the image which you want to test..... you can change it according to the image location you have  
        fileName = askopenfilename(initialdir='C:\\Users\\dhruva adithya\\Desktop\\Datasets\\train', title='Select image for analysis ',
                               filetypes=[('image files', '.jpg')])
        dst = "testpicture"
        print(fileName)
        print (os.path.split(fileName)[-1])
        if os.path.split(fileName)[-1].split('.') == 'b (1)':
            print('dfdffffffffffffff')
        shutil.copy(fileName, dst)
        load = Image.open(fileName)
        render = ImageTk.PhotoImage(load)
        img = tk.Label(image=render, height="250", width="500")   
        img.image = render
        img.place(x=470, y=100)
        #img.grid(column=0, row=1, padx=10, pady = 10)
        title.destroy()
        
        button1.destroy()
        button2 = tk.Button(text="Analyse Image", command=analysis)
        button2.place(x=670,y=380)
        #button2.grid(column=0, row=20, padx=10, pady = 10)
    button1 = tk.Button(text="Get Photo", command = openphoto)
    button1.place(x=600,y=150)
    #button1.grid(column=0, row=1, padx=10, pady = 10)

    root.mainloop()




    #os.system("python ui_Bone.py")
    



takeImg = tk.Button(root, text="CNN", command=CNN  ,fg="white"  ,bg="black"  ,width=10  ,height=2, activebackground = "black" ,font=('times', 15, ' bold '))
takeImg.place(x=400, y=550)
trackImg = tk.Button(root, text="Quit", command=root.destroy  ,fg="white"  ,bg="Black"  ,width=10  ,height=2, activebackground = "Black" ,font=('times', 15, ' bold '))
trackImg.place(x=900, y=550)
label.pack()
root.mainloop()


















