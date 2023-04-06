import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from time import sleep
#Citiți și afișați, în aceeași fereastră, imaginile amprentelor unui deget dintr-una din cele patru baze de date DB1÷DB4
def cerinta1(folder):
    images_db2=[ ]

    fig = plt.figure(figsize=(10,8))
    rows=2
    columns=4
    count=1
    for img in glob.glob("./"+folder+"/110*.tif"):
        images_db2.append(cv2.imread(img) )
        fig.add_subplot(rows, columns, count)
        plt.imshow(images_db2[count-1])
        count=count+1
        plt.title(img)
    plt.show()

#Citiți și afișați, în aceeași fereastră, câte o imagine a amprentelor primelor patru degete dintr-una din cele patru baze de date DB1÷DB4
def cerinta2(folder):
    fig = plt.figure(figsize=(10,8))
    rows=1
    columns=4
    for i in range(1,5,1):
        title="./"+folder+"/10"+str(i)+"_1.tif"
        img=cv2.imread(title)
        fig.add_subplot(rows,columns,i)
        plt.imshow(img)
        plt.title(title)
    plt.show()    

#a) reprezentați histogramele imaginilor
def cerinta3_a(title):
    fig = plt.figure(figsize=(20,20))
    img=cv2.imread(title)
    hist=cv2.calcHist([img],[0],None,[256],[0,256])
    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(title)
    fig.add_subplot(1, 2, 2)
    plt.hist(img.ravel(),256,[0,256])
    plt.title("Histogram")
    plt.show()
    
    plt.close()

#b) calculați și afișați valorile minime și maxime ale pixelilor pentru fiecare imagine
#aducem imaginile in intervalul 0,1 pentru ca provin din baze de date diferite s au intesitati ale pixelilor diferite
def cerinta3_b(title):
    img=cv2.imread(title)
    print("Max pixel value is: " + str(img.reshape((img.shape[0]*img.shape[1],3)).max(axis=0)))
    print("Min pixel value is: " + str(img.reshape((img.shape[0]*img.shape[1],3)).min(axis=0)))
 
    #metoda mai naspa
    #img=cv2.imread(title).tolist()
    #flat_img=[item for sublist in img for item in sublist]
    #print("Max pixel value is: "+str(max(flat_img)))
    
#normalizați imaginile astfel încât valorile pixelilor să varieze în intervalul [0, 1];
def  cerinta3_c(title):
    fig = plt.figure(figsize=(10,10))
    img=cv2.imread(title)
    #img_normalized = cv2.normalize(img, None, 0,1.0,cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    max_pixel=img.max()
    min_pixel=img.min()
    print(max_pixel)
    img_normalized=img.copy()
    img_normalized=(img_normalized-min_pixel)/(max_pixel-min_pixel)
    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(title)
    fig.add_subplot(1, 2, 2)
    plt.imshow(img_normalized)
    plt.title(title+ " Normalized")
    plt.show()
    plt.close()

#calculați media și abaterea (deviația) standard a valorilor pixelilor pentru fiecare imagine;
#scadem media si impartim la std
def cerinta3_d(title):
    img=cv2.imread(title)
    img=np.reshape(img[:,:,0],-1)
    print("Mean value of pixels is: "+str(np.mean(img)))
    print("Std value of pixels is: "+str(np.std(img)))
          
    
#normalizați imaginile astfel încât media valorilor pixelilor să fie 0, iar deviația standard 1
def cerinta3_e(title):
    img=Image.open(title)
    imgTensor=T.ToTensor()(img)
    transform = T.Normalize((0.0), 
                            (1.0))
  
    normalized_imgTensor = transform(imgTensor)
    normalized_img = T.ToPILImage()(normalized_imgTensor)
    print(normalized_img)
    normalized_img.show()
    normalized_img.close()


def cerinta3(imgs):
    for img in imgs:
        cerinta3_a(img)
        
        cerinta3_b(img)
        cerinta3_c(img)
        cerinta3_d(img)
        cerinta3_e(img)

#cerinta1("DB2_B")
#cerinta2("DB2_B")

titles =["./DB1_B/101_1.tif","./DB2_B/101_1.tif","./DB3_B/101_1.tif"]
for i in range(3):
    cerinta3_c(titles[i])
