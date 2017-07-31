#-*-coding:utf8-*-
__author__ = '何斌'
__datetime__='2017-7-15'

from dataSet import DataSet
from keras.models import Sequential,load_model
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten,Dropout
import numpy as np
import os
 
datasets_path='G:\Python/FaceRecognition\datasets'  #数据集路径 
read_save=True                                             #训练模型是否保存

#建立一个基于CNN的人脸识别模型
class Model(object):
    FILE_PATH = "model\model.ckpt"                             #模型进行存储和读取的地方
    IMAGE_SIZE = 128                                           #模型接受的人脸图片一定是128*128    

    def __init__(self,read_save=True):
        self.model = None
        self.read_save=read_save

    #读取实例化后的DataSet类作为进行训练的数据源
    def read_trainData(self,dataset):
        self.dataset = dataset

    #建立一个CNN模型，一层卷积、一层池化、一层卷积、一层池化、抹平之后进行全链接、最后进行分类
    def build_model(self):
        self.model = Sequential()
        self.model.add(
            Convolution2D(
                kernel_size=(5, 5),
                # dim_ordering='th',
                input_shape=self.dataset.X_train.shape[1:],
                padding='same',
                filters=32
            )
        )

        self.model.add(Activation('relu'))
        self.model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same'
            )
        )
        
        self.model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        

        self.model.add(Flatten())
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        

        self.model.add(Dense(self.dataset.num_classes))
        self.model.add(Activation('softmax'))
        self.model.summary()

    #进行模型训练的函数，具体的optimizer、loss可以进行不同选择
    def train_model(self):
        self.model.compile(
            optimizer='RMSprop',  #有很多可选的optimizer，例如RMSprop,Adagrad，你也可以试试哪个好，我个人感觉差异不大
            loss='squared_hinge', #'categorical_crossentropy',  #你可以选用squared_hinge作为loss看看哪个好
            metrics=['accuracy'])
        #epochs、batch_size为可调的参数，epochs为训练多少轮、batch_size为每次训练多少个样本
        self.model.fit(self.dataset.X_train,self.dataset.Y_train,epochs=8,batch_size=20)

    def evaluate_model(self):
        print('\nTesting---------------')
        loss, accuracy = self.model.evaluate(self.dataset.X_test, self.dataset.Y_test)

        print('test loss;', loss)
        print('test accuracy:', accuracy)

    def save(self, file_path=FILE_PATH):
        if not os.path.exists('model/'):
            os.mkdir('model/')
        self.model.save(file_path)
        print('Model Saved.')

    def load(self, file_path=FILE_PATH):
        if self.read_save and os.path.exists(file_path):
            self.model = load_model(file_path)
        else:
            self.build_model()
        print('Model Loaded.')
        

    #需要确保输入的img得是灰化之后（channel =1 )且 大小为IMAGE_SIZE的人脸图片
    def predict(self,img):
        img = img.reshape((1, 1, self.IMAGE_SIZE, self.IMAGE_SIZE))
        img = img.astype('float32')
        img = img/255.0
        result = self.model.predict_proba(img)  #测算一下该img属于某个label的概率
        max_index = np.argmax(result)#找出概率最高的
        return max_index,result[0][max_index] #第一个参数为概率最高的label的index,第二个参数为对应概率


if __name__ == '__main__':
    dataset = DataSet(datasets_path)
    dataset.check()
    model = Model(read_save=read_save)
    model.read_trainData(dataset)
    model.load()
    model.train_model()
    model.evaluate_model()
    model.save()














