import os
from PIL import Image

pic_dir = r"G:\Python\faceRecognition-master\images\mix_pic"
save_dir=r"G:\Python\faceRecognition-master\images\mix_pic1"
if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
index=1         #文件名
for filename in os.listdir(path=pic_dir):
        pic_path = os.path.join(pic_dir, filename)
        print (pic_path)
        img = Image.open(pic_path)
##        new_size = tuple([32,32])
##        new_img = img.resize( new_size)
        filename1="1_{}.jpg".format(index)
        new_name = os.path.join(save_dir, filename1)
        img.save(new_name)
        index+=1

