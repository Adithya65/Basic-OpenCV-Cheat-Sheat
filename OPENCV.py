#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2 as cv


### image capture

# In[ ]:


image =cv.imread('dd.jpg')  #return image as matrix


# In[ ]:


cv.imshow('test1',image)  #("title",image name)
cv.waitKey(0)


# # video capture
# 

# In[ ]:


video1=cv.VideoCapture(0)
while True:
    isTrue,frame =video1.read()
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    cv.imshow("video_title",gray)
    cv.imshow("original",frame)
    if cv.waitKey(1) & 0xFF ==ord('d'):
        break
video1.release()
cv.destroyAllWindows()


# # Resizing and Rescaling

# In[ ]:


def rescaleframe(frame,scale =0.75):
    width=int(frame.shape[1]*scale)
    height =int(frame.shape[0]*scale)
    dimensions=(width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)


video1=cv.VideoCapture(0)
    
while True:
    isTrue, frame =video1.read()
    frame_resized =rescaleframe(frame)
    cv.imshow("video_title",frame)
    cv.imshow("videoresized",frame_resized)
    if cv.waitKey(20) & 0xFF ==ord('d'):
        break
video1.release()
cv.destroyAllWindows()


# In[ ]:


def rescaleframe(frame,scale =.1):
    width=int(frame.shape[1]*scale)
    height =int(frame.shape[0]*scale)
    dimensions=(width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

img=cv.imread("sd.jpg")
resizedimage=rescaleframe(img)
cv.imshow("resized",resizedimage)
cv.waitKey(0)
cv.destroyAllWindows()


# In[ ]:


import cv2 as cv 
import numpy as np
img =cv.imread("dd.jpg")


# In[ ]:


blank=np.zeros((500,500,3),dtype='uint8')
blank[200:400,300:600]=0,0,255
cv.imshow("blank",blank)
cv.waitKey(0)
cv.destroyAllWindows()


# # Rectangle

# In[ ]:


test1=np.zeros((500,500,3),dtype='uint8')
cv.rectangle(tets1,(0,0),(250,500),(0,255,0),thickness =cv.FILLED)
#cv.rectangle(test1,(0,0),(blank.shape[0]//2,blank.shape[1]//2),(0,0,255),thickness=-1)
cv.imshow('rectangle',blank)
cv.waitKey(0)
cv.destroyAllWindows()


# # Circle

# In[ ]:


test2=np.zeros((500,500,3),dtype='uint8')
cv.circle(test2,(400,250),10,thickness=-1,color=(0,0,255))
cv.imshow("circle",test2)
cv.waitKey(0)
cv.destroyAllWindows()


# # Line

# In[ ]:


test3=np.zeros((500,500,3),dtype='uint8')

cv.line(test3,(0,0),(250,250),(0,255,0),thickness=4)
cv.line(test3,(400,400),(300,300),(100,28,90),thickness=4) #2nd line


cv.imshow("line1",test3)

cv.waitKey(0)
cv.destroyAllWindows()


# # Text

# In[ ]:


test4=np.zeros((500,500,3),dtype='uint8')
cv.putText(test4,"hello World",(225,225),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,0),thickness=3)
cv.imshow("text",test4)
cv.waitKey(0)
cv.destroyAllWindows()


# # colour changing to gray

# In[ ]:


import cv2 as cv 
import numpy as np
img =cv.imread("dd.jpg")
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("gray",gray)
cv.imshow("ORIGINAL",img)
cv.waitKey(0)
cv.destroyAllWindows()


# # Blur
# 

# In[ ]:


import cv2 as cv 
import numpy as np
img =cv.imread("dd.jpg")
blur=cv.GaussianBlur(img,(5,5),cv.BORDER_DEFAULT)
cv.imshow("blur",blur)
cv.imshow("ORIGINAL",img)
cv.waitKey(0)
cv.destroyAllWindows()


# # Edges
# 

# In[ ]:


import cv2 as cv 
import numpy as np
img =cv.imread("dd.jpg")
canny=cv.Canny(img,100,180)
cv.imshow("canny",canny)
cv.imshow("ORIGINAL",img)
cv.waitKey(0)
cv.destroyAllWindows()


# In[ ]:





# # Dilate

# In[ ]:


import cv2 as cv 
import numpy as np
img =cv.imread("dd.jpg")
canny=cv.Canny(img,100,180)
dilated=cv.dilate(canny,(7,7),iterations=5)
cv.imshow("canny",canny)
cv.imshow("dilated",dilated)
cv.waitKey(0)
cv.destroyAllWindows()


# # Erode

# In[ ]:


import cv2 as cv
img=cv.imread("dd.jpg")
canny=cv.Canny(img,100,180)
dialated=cv.dilate(canny,(3,3),iterations=1)
eroded=cv.erode(dialated,(7,7),iterations=3)
cv.imshow("canny",canny)
cv.imshow("Erode",eroded)
cv.imshow("dilated",dialated)
cv.waitKey(0)
cv.destroyAllWindows()


# # Resize

# In[ ]:


resized=cv.resize(img,(500,500),interpolation=cv.INTER_CUBIC)
cv.imshow("resized",resized)
cv.waitKey(0)
cv.destroyAllWindows()


# # Crop

# In[ ]:


cropped=img[50:200,200:400]
cv.imshow("cropped",cropped)
cv.waitKey(0)
cv.destroyAllWindows()


# 
# import cv2 as cv
# img=cv.imread("butterfly.jpeg")
# cv.imshow("norma",img)
# 
# gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY) 
# cv.imshow("gray",gray)
# ret, thresh=cv.threshold(gray,100,255,cv.THRESH_BINARY)
# cv.imshow("thresh",thresh)
# gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY) ##some error
# cv.imshow("gray",gray)
# canny=cv.Canny(img,125,175)
# contours=[]
# cv.imshow("CANNY",canny)
# contours =cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
# print(len(contours))
# cv.waitKey(0)
# cv.destroyAllWindows()
# 
# 

# # Thresholding

# In[ ]:


import cv2 as cv
img=cv.imread("butterfly.jpeg")
cv.imshow("norma",img)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY) 
cv.imshow("gray",gray)
ret,thresh=cv.threshold(gray,100,255,cv.THRESH_BINARY)
cv.imshow("thresh",thresh)
cv.waitKey(0)
cv.destroyAllWindows()


# In[ ]:


import numpy as np
blank=np.zeros(img.shape[:2],dtype='uint8')
cv.imshow("balnk",blank)
cv.drawContours(blank,contours,-1,(0,0,255),2)
cv.imshow("blank",blank)
cv.waitKey(0)
cv.destroyAllWindows()


# # HSV, LAB, RGB

# In[ ]:


img=cv.imread("dd.jpg")
hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV) #hue saturation value
cv.imshow("gray",hsv)

lab=cv.cvtColor(img,cv.COLOR_BGR2LAB) #L=light,A-Green(changes to black) to Magenta(changes to white),
#B-(Blue will be cahnged to black) to Yellow(and yellow to white),which indeed converts other color gradients.
cv.imshow("lab",lab)

rgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
cv.imshow("rgb",rgb)

cv.waitKey(0)
cv.destroyAllWindows()


# # BGR2RGB USING MATPLOTLIB

# In[ ]:


import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()


# In[ ]:


vid=cv.VideoCapture(0)
while True:
    isTrue,frame=vid.read()
    rgb=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    cv.imshow("video",rgb)
    if cv.waitKey(1) & 0xFF==ord('d') :
        break
vid.release()
cv.destroyAllWindows()
    


# In[1]:


import cv2 as cv
img=cv.imread("butterfly.jpeg")
cv.imshow("norma",img)
b,g,r=cv.split(img)
cv.imshow('blue',b)
cv.imshow('green',g)
cv.imshow('red',r)
cv.waitKey(0)
cv.destroyAllWindows()


# In[9]:


import numpy as np
blank=np.zeros(img.shape[:2],dtype='uint8')
b,g,r=cv.split(img)

blue=cv.merge([b,blank,blank])
green=cv.merge([blank,g,blank])
red=cv.merge([blank,blank,r])


cv.imshow("Red",red)
cv.imshow("blue",blue)
cv.imshow('green',green)
cv.waitKey(0)
cv.destroyAllWindows()


# In[ ]:




