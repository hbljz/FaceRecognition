import os,cv2
import random
import skimage.data
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from read_data import read_file

#读取一个文件夹，该文件夹的子文件夹为标签，子文件夹里存放对应的图片
#并返回两个list（images，labels）
# def load_data(data_dir):
#     #读取文件夹里的图片放在一个列表中
#     directories = [d for d in os.listdir(data_dir) 
#                    if os.path.isdir(os.path.join(data_dir, d))]               
#     #两个list存放图片和标签
#     labels = []
#     images = []

#     #遍历列表提取出图片对应的标签，分别存放在labels和images两个列表
#     for d in directories:
#         label_dir = os.path.join(data_dir, d)
#         file_names = [os.path.join(label_dir, f) 
#                       for f in os.listdir(label_dir) if f.endswith(".jpg")]
#         for f in file_names:
#             images.append(skimage.data.imread(f))
#             labels.append(int(d))
#     return images, labels

#定义一个函数，用于初始化所有的权值 W
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
#定义一个函数，用于初始化所有的偏置项 b
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
#定义一个函数，用于构建卷积层
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#定义一个函数，用于构建池化层
def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

# 开始读取存放训练图片的文件夹.
# train_data_dir = "G:\Python/egg/datasets/Training"
# test_data_dir = "G:\Python/egg/datasets/Testing"
# images, labels = load_data(train_data_dir)

#将两个list转化为可运算的数组
# images_a = np.array(images)
# labels_a = np.array(labels)
# print("labels: ", labels_a.shape, "\nimages: ", images_a.shape)#查看两个数组的结构

# #创建图表
# graph = tf.Graph()
# with graph.as_default():
# 设置两个占位符为输入值
images_ph = tf.placeholder(tf.float32, [None, 128, 128, 1])
labels_ph = tf.placeholder(tf.int32, [None])

# # 将图片转化为一维的数组，压平图片
# # To: [None, height * width * channels] == [None, 3072]
# images_flat = tf.contrib.layers.flatten(images_ph)
# 
#第一个卷积层
W_conv1 = weight_variable([5, 5, 1, 32])      
b_conv1 = bias_variable([32])       
h_conv1 = tf.nn.relu(conv2d(images_ph, W_conv1) + b_conv1)
#第一个池化层
h_pool1 = max_pool(h_conv1)

#第二个卷积层
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

#第二个池化层      
h_pool2 = max_pool(h_conv2)

W_fc1 = weight_variable([32*32*64, 16])
b_fc1 = bias_variable([16])
h_pool2_flat = tf.reshape(h_pool2, [-1, 32*32*64])              #reshape成向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)    #第一个全连接层 

# 全连接池层 
logits = tf.contrib.layers.fully_connected(h_fc1,8, tf.nn.relu)

keep_prob = tf.placeholder(tf.float32) 
logits1 = tf.nn.dropout(logits, keep_prob) 

# 将连接层结果的最大值为预测值
# Shape [None],  a 1D vector of length == batch_size.
predicted_labels = tf.argmax(logits1,1)
    
# 定义交叉熵损失函数. 
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits1, labels_ph))

# 使用梯度下降训练
train = tf.train.AdamOptimizer(2e-4).minimize(loss)
#train = tf.train.AdagradOptimizer(2e-4).minimize(loss)
    

# print("images_flat: ", images_flat)     #查看压平图片后的数组结构
print("logits1: ", logits1)       #查看全连接池结构
print("loss: ", loss)           #查看损失变量结构
print("predicted_labels: ", predicted_labels)   #查看预测数组的结构


saver = tf.train.Saver()

#判断模型保存路径是否存在，不存在就创建
if not os.path.exists('model1/'):
    os.mkdir('model1/')

#初始化
session = tf.Session()
if os.path.exists('model1/checkpoint'): #判断模型是否存在
    saver.restore(session, 'model1/model1.ckpt') #存在就从模型中恢复变量
else:
    init = tf.global_variables_initializer() #不存在就初始化变量
    session.run(init)

datasets_path='G:\Python/FaceRecognition\datasets'
imgs,labels,counter = read_file(datasets_path)
imgs=imgs.reshape(imgs.shape[0],128,128,1)
# labels=labels.reshape(labels.shape[0])
# for i in range(10):
#     train_value, loss_value = session.run([train, loss], 
#                                 feed_dict={images_ph: imgs, labels_ph: labels,keep_prob:0.8})
#     if(i%2==0): 
#         save_path = saver.save(session, 'model1/model1.ckpt') #保存模型到tmp/model.ckpt，注意一定要有一层文件夹，否则保存不成功！！！
#         print("模型保存：%s 当前训练损失：%s"%(save_path, loss_value))
        

# # 随机抽二十张图片测试
# sample_indexes = random.sample(range(len(images)), 20)#20表示我们需要测试的图片数量
# sample_images = [images[i] for i in sample_indexes]
# sample_labels = [labels[i] for i in sample_indexes]

#开始运行测试得到测试结果
img = cv2.imread(r"G:/Python/FaceRecognition/images/hb/frmaes_18.jpg")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,1.2,5)
for (x, y, w, h) in faces:
    face = cv2.resize(gray[y:(y + h), x:(x + w)], (128, 128))
face=face.reshape(-1,128,128,1)
predicted = session.run([predicted_labels], 
                        feed_dict={images_ph: face,keep_prob:0.8})[0]
# print("truth:",sample_labels)
print("prediction:",predicted)


#用matplotlib将结果可视化.
# fig = plt.figure(figsize=(10, 10))
# for i in range(len(sample_images)):
#     truth = sample_labels[i]
#     prediction = predicted[i]
#     plt.subplot(10, 2,i+1)
#     plt.axis('off')
#     color='green' if truth == prediction else 'red'
#     plt.text(40, 10, "Truth:         {0}\nPrediction:  {1}".format(truth, prediction), 
#              fontsize=12, color=color)
#     plt.imshow(sample_images[i])
# plt.show()