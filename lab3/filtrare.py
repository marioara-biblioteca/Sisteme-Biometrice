import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def cerinta1(img):
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()


def cerinta2_gray(img,i):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_fft=np.fft.fftshift(np.fft.fft2(gray))
    
    gray_fft[235:240, :230] = i
    gray_fft[235:240,-230:] = i
    
    fig,ax=plt.subplots(1,3)
    ax[0].imshow(gray,cmap='gray')
    ax[0].set_title("Gray Image",fontsize = 10)
    ax[1].imshow(np.log(abs(gray_fft)),cmap='gray')
    ax[1].set_title("Masked Fourier",fontsize = 10)
    ax[2].imshow(abs(np.fft.ifft2(gray_fft)), 
                        cmap='gray')
    ax[2].set_title('Transformed Greyscale Image', 
                     fontsize = 10)
    plt.show()
def cerinta2_color(img):
    transformed_channels=[]
    for i in  range(3): #RGB
        rgb_fft=np.fft.fftshift(np.fft.fft2((img[:,:,i])))
        rgb_fft[235:240, :230]=1
        rgb_fft[235:240,-230:]=1
        transformed_channels.append(abs(np.fft.ifft2(rgb_fft)))
    final_image = np.dstack([transformed_channels[0].astype(int), 
                             transformed_channels[1].astype(int), 
                             transformed_channels[2].astype(int)])
    fig, ax = plt.subplots(1, 2, figsize=(17,12))
    ax[0].imshow(img)
    ax[0].set_title('Original Image', fontsize = 15)
    ax[0].set_axis_off()
    
    ax[1].imshow(final_image)
    ax[1].set_title('Transformed Image', fontsize = 15)
    ax[1].set_axis_off()
    
    plt.show()

def cerinta3(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    treshs = [cv.THRESH_BINARY,cv.THRESH_BINARY_INV,cv.THRESH_TRUNC,cv.THRESH_TOZERO,cv.THRESH_TOZERO_INV]
    titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    imgs=[img]
    for thr in treshs:
        ret,bin_img = cv.threshold(img,127,255,thr)
        imgs.append(bin_img)
    for i in range(len(treshs)+1):
        plt.subplot(2,3,i+1),plt.imshow(imgs[i],'gray',vmin=0,vmax=255)
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
   
file='../lab1/DB1_B/101_1.tif'
# file='./duck.jpg'
img=cv.imread(file)
	
#cerinta1(img)

#In the image we can see two very clear distortions. The white vertical and horizontal lines refer to the sharp horizontal and vertical elements of the image. Let us see what happens if we mask one of them.

# plt.figure(num=None, figsize=(8, 6), dpi=80)
# plt.imshow(np.log(abs(np.fft.fftshift(np.fft.fft2(cv.cvtColor(img, cv.COLOR_BGR2GRAY))))), cmap='gray');
# plt.show()

# cerinta2_color(img)

cerinta3(img)