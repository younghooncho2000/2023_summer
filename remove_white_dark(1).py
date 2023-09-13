#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#상하좌우 검은 배경 제거하기
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
image_path = "/home/save/Downloads/tmp_img" ## tmp_img 폴더 안에 000001, 000002 , ... 있고, 000001 안에 이미지들 있어요

def img_show(title='image', img=None, figsize=(8 ,5)): #show image
    plt.figure(figsize=figsize)
 
    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []
 
        for i in range(len(img)):
                titles.append(title)
 
        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
 
            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
 
        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()

list1= []
#check black backgrounds
def cond1(height, width): 
    if image[height, width][0] < 10 and image[height, width][1] < 10 and image[height, width][2] < 10:
        return True
#check white backgrounds
def cond2(height, width):
    if image[height, width][0] > 245 and image[height, width][1] > 245 and image[height, width][2] > 245:
        return True

for image_path_per_folder in os.listdir(image_path):
    for img_path in os.listdir(image_path + '/' +image_path_per_folder):
        real_image = cv2.imread(image_path + '/' + image_path_per_folder + '/' + img_path, cv2.IMREAD_COLOR)
        image = cv2.imread(image_path + '/' + image_path_per_folder + '/' + img_path, cv2.IMREAD_COLOR)
        (h, w) = image.shape[:2]
        left_x = 0
        right_x = w
        top_h = 0
        bottom_h = h
        #상하좌우 확인 / 가장자리부터 체크
        for a in range(0, int(w/2 - 1)):
            cnt = 0
            for n in range(0, h) :
                if cond1(n,a) or cond2(n,a):
                    cnt += 1

            if cnt > h-10:
                left_x = a

            else:
                break
                        
        for b in reversed(range(int(w/2 + 1), w)):
            cnt = 0
            for n in range(0, h):
                if cond1(n,b) or cond2(n,b):
                    cnt += 1

            if cnt > h-10:
                right_x = b

                
            else:

                break
                
        for c in range(int(h/3 - 1)):
            cnt = 0
            for n in range(0,w):
                if cond1(c,n) or cond2(c,n):
                    cnt += 1

            if cnt > w-10:
                top_h = c

            else:
                break

        for d in reversed(range(int(h/3 + 1), h)):
            cnt = 0
            for n in range(0,w):
                if cond1(d,n) or cond2(d,n):
                    cnt += 1
            if cnt > w-10:
                bottom_h = d
            else:
                break
                


        sliced_img = image[top_h:bottom_h, left_x:right_x]
        if sliced_img.shape[0] > 100 and sliced_img.shape[1] > 100:
            cv2.imwrite(image_path + '/' +image_path_per_folder+'/'+img_path , sliced_img) 

