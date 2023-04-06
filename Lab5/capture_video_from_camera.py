import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import mtcnn
print(mtcnn.__version__)
cap = cv2.VideoCapture(0)
# width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
# fps =  cap.get(cv2.CAP_PROP_FPS)
# print('Width,Height')
# print(width,height)
# print('FPS')
# print(fps)

detector = mtcnn.MTCNN()
frontalface=cv2.CascadeClassifier('lbpcascade_frontalface.xml')
haar=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

def draw(result,i,img):
    bounding_box = result[i]['box']
    keypoints = result[i]['keypoints']
    cv2.rectangle(img,
                (bounding_box[0], bounding_box[1]),
                (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                (0,155,255),
                2)

    # cv2.circle(img,(keypoints['left_eye']), 2, (255,0,0), 2)
    # cv2.circle(img,(keypoints['right_eye']), 2, (255,0,0), 2)
    # cv2.circle(img,(keypoints['nose']), 2, (255,0,0), 2)
    # cv2.circle(img,(keypoints['mouth_left']), 2, (255,0,0), 2)
    # cv2.circle(img,(keypoints['mouth_right']), 2, (255,0,0), 2)

while(True):
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame=cv2.resize(frame,dsize=None,fx=0.5,fy=0.5)
    # Our operations on the frame come here

    #MTCNN
    result=detector.detect_faces(frame)
    for i in range(len(result)):
        draw(result,i,frame)

    gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)    
    #frontalface
    faces1=frontalface.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3)
    for (x, y, w, h) in faces1:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    # #HAAR  
    faces2=haar.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3)
    for (x, y, w, h) in faces2:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
