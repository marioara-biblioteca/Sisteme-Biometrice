from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

img=cv2.imread('./imageFile2.jpg') 

def isgray(img):
    if len(img.shape) < 3: return True
    if img.shape[2]  == 1: return True
    b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    if (b==g).all() and (b==r).all(): return True
    return False

def cerinta1(img):
    print(f"Imaginea e reprezentata cu pixelii sub forma: {img.dtype}, {img[0][0]}")
    if isgray(img):
        print("Gray")
    else:
        print("RGB")
    print(f"Dimensiunile imaginii sunt: {img.ndim},{img.shape}.")
    fig = plt.figure(figsize=(5,5))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

def detect_face(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects
def  detect_eyes(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=(5, 5))
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects
def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
def cerinta2(img):
    
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    rects = detect_face(gray, face_cascade)
    vis = img.copy()
    
    draw_rects(vis, rects, (0, 255, 0))
    for x1, y1, x2, y2 in rects:
        roi = gray[y1:y2, x1:x2]
        vis_roi = vis[y1:y2, x1:x2]
        subrects = detect_eyes(roi.copy(), eye_cascade)
        draw_rects(vis_roi, subrects, (255, 0, 0))
    fig = plt.figure(figsize=(5,5))
    plt.imshow( cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.show()
#cerinta1(img)
cerinta2(img)