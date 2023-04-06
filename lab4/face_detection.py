
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import time
def cerinta1(img,model):
    lenna_gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #fig,axs=plt.subplots(1,2)
    #axs[0].imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
    #axs[1].imshow(lenna_gray,cmap='gray')
    #plt.show()
    faces=model.detectMultiScale(lenna_gray, scaleFactor=1.05, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
    plt.show()
def all(images, model1,model2):
    fig,axs=plt.subplots(len(images),2)
   
    for i in range(len(images)):
        image=cv.imread(images[i])
        
        img1=image.copy()
        img2=image.copy()
       
        gray_img1=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
        start1= time.time()
        faces1=model1.detectMultiScale(gray_img1, scaleFactor=1.05, minNeighbors=3)
        stop1=time.time()
       
        gray_img2=cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
        start2=time.time()
        faces2=model2.detectMultiScale(gray_img2, scaleFactor=1.05, minNeighbors=3)
        stop2=time.time()
        print(i,"LBP detection: "+str(stop1-start1) + "  Haar detection: "+str(stop2-start2))
        for (x, y, w, h) in faces1:
            cv.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        for (x, y, w, h) in faces2:
            cv.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 0), 2)
        axs[i][0].imshow(cv.cvtColor(img1,cv.COLOR_BGR2RGB))
        axs[i][0].margins(x=0)
        axs[i][1].margins(x=0)
        axs[i][0].set_title(str(stop1-start1),fontdict={'fontsize': 7, 'fontweight': 'medium'})
        axs[i][1].imshow(cv.cvtColor(img2,cv.COLOR_BGR2RGB))
        axs[i][1].set_title(str(stop2-start2),fontdict={'fontsize': 7, 'fontweight': 'medium'})
   
    plt.show()
lenna=cv.imread('./lena_color_copy.png')
frontalface=cv.CascadeClassifier('lbpcascade_frontalface.xml')
haar=cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
images=['lena_color_copy.png','Friends.jpg','oscars-selfie_copy.jpg','crowd.jpg','biggest-selfie.png']
# cerinta1(lenna,frontalface)
all(images,frontalface,haar)