#-*-coding:utf8-*-
__author__ = '何斌'
__datetime__='2017-7-15'

import os
import cv2


#源图片文件目录
Img_path=""
Suffix=['.jpg','.PNG','.png']

#根据输入的文件夹绝对路径，将文件夹下所有指定suffix(后缀名)读取存入一个list(resultArray)
#该文件夹的名字存入list的第一个元素
def readAllImg(path,*suffix):
    try:
        s = os.listdir(path)
        resultArray = []
        img_name_list=[]
        fileName = os.path.basename(path)   #文件夹名称，如basename("a\b\c")结果为c
        # resultArray.append(fileName)

        for item in s:
            if endwith(item, suffix):
                document = os.path.join(path, item)
                img = cv2.imread(document)
                img_name_list.append(item)
                resultArray.append(img)

    except IOError:
        print("Error")

    else:
        print("读取图片成功...")
    return resultArray,img_name_list

#输入一个字符串s一个标签，对这个字符串的后缀和标签进行匹配
def endwith(s,*endstring):
   results = map(s.endswith,endstring)
   if True in results:
       return True
   else:
       return False

# if __name__ == '__main__':
  # results = readAllImg(Img_path,Suffix)
  # print(results[0])
  # cv2.namedWindow("Image")
  # cv2.imshow("Image", result[1])
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()