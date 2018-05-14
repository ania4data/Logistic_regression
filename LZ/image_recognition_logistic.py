
# coding: utf-8

# In[36]:


import numpy as np
import cv2
import imageio as io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


imgcv2=cv2.imread('galaxy.jpg',0)   # 1 as is    # 0  blackwhite  -1 alpha channel transparency .color image is three dimensional matrix
img2=cv2.imread('cat.jpg')

cv2.imshow('Cats',img2)
cv2.waitKey(5000)
cv2.destroyAllWindows()


np.shape(img2)
#img=io.imread('galaxy.jpg')
#imageio.imshow(img)
#img3=plt.imread('cat.jpg')
#plt.imshow(img3)


# In[47]:


count=0

label_map=['Anger','Disgust','Fear','Happy','Sad','Surprise','Neutral']

for line in open('fer2013_cleaned.txt'):
    count=count+1
    if(count<20):
        
        list_=line.replace(',',' ').split(' ')
        expr=int(list_[0])
        txt=list_[len(list_)-1]
        image_vec=np.array((list_[1:len(list_)-1]))
        image_vec_i=[int(i) for i in image_vec]
        image_vec_i=np.array(image_vec_i)
        len_N=int(np.sqrt(np.shape(image_vec_i)[0]))
        img=image_vec_i.reshape(len_N,len_N)
        img_norm=img/(np.max(img))
        plt.figure(figsize=(4,4))
        plt.imshow(img,cmap='gray')
        plt.title(label_map[expr])
        plt.show()
        
        plt.figure(figsize=(4,4))
        plt.imshow(img_norm,cmap='gray')
        plt.title(label_map[expr])
        plt.show()        
        
    else:
        break
        
#print(expr)
#print(len(list_))
#print(txt) 
#print(image_vec)
#print(len(image_vec))
#print(image_vec_i)
#print(np.shape(image_vec_i))
#print(len_N)


# In[34]:


#for line in open('first_line.txt'):
#    
#    list_=line.replace(',',' ').split(' ')
#    expr=int(list_[0])
#    txt=list_[len(list_)-1]
#    image_vec=np.array((list_[1:len(list_)-1]))
#    image_vec_i=[int(i) for i in image_vec]
#    image_vec_i=np.array(image_vec_i)
#    len_N=int(np.sqrt(np.shape(image_vec_i)[0]))
#    img=image_vec_i.reshape(len_N,len_N)

