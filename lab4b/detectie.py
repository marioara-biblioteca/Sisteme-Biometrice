
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import mtcnn
import cv2
import numpy as np
import matplotlib.pyplot as plt



def draw(result,i):
    print(result)
    bounding_box = result[i]['box']
    keypoints = result[i]['keypoints']
    cv2.rectangle(img,
                (bounding_box[0], bounding_box[1]),
                (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                (0,155,255),
                2)

    cv2.circle(img,(keypoints['left_eye']), 2, (255,0,0), 2)
    cv2.circle(img,(keypoints['right_eye']), 2, (255,0,0), 2)
    cv2.circle(img,(keypoints['nose']), 2, (255,0,0), 2)
    cv2.circle(img,(keypoints['mouth_left']), 2, (255,0,0), 2)
    cv2.circle(img,(keypoints['mouth_right']), 2, (255,0,0), 2)

print(mtcnn.__version__)
img = cv2.cvtColor(cv2.imread("test1.jpg"), cv2.COLOR_BGR2RGB)
detector = mtcnn.MTCNN()
result=detector.detect_faces(img)

for i in range(len(result)):
    draw(result,i)

plt.imshow(img)
plt.show()