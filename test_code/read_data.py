#-*-coding:utf8-*-
__author__ = '何斌'
__datetime__='2017-7-15'

import os
import cv2
import numpy as np
from read_img import endwith

#输入一个文件路径，对其下的每个文件夹下的图片读取，并对每个文件夹给一个不同的Label
#返回一个img的list,返回一个对应label的list,返回共有几个文件夹即有几种label（dir_counter）
def read_file(path):
    img_list = []       #全部图片list
    label_list = []     #标签列表
    dir_counter = 0     #子文件夹个数
    IMG_SIZE = 128

    #对路径下的所有子文件夹中的所有jpg文件进行读取并存入到一个list中
    for child_dir in os.listdir(path):
         child_path = os.path.join(path, child_dir)
         #进入子文件夹读取图片
         for dir_image in  os.listdir(child_path):
             if endwith(dir_image,'jpg'):
                img = cv2.imread(os.path.join(child_path, dir_image))
                resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                recolored_img = cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
                img_list.append(recolored_img)
                label_list.append(dir_counter)
        #子文件夹个数加1
         dir_counter += 1
    #转成np.array的格式并返回
    img_list = np.array(img_list)

    return img_list,label_list,dir_counter

#读取训练数据集的文件夹，把他们的名字返回给一个list
def read_name_list(path):
    name_list = []
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list



if __name__ == '__main__':
    img_list,label_lsit,counter = read_file('')
    print (counter)


