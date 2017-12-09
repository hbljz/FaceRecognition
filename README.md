# faceRecognition
利用OpenCV、CNN进行人脸识别(开发环境：Python3.5+Opencv3.2.0+Keras2.0.6)


__author__ = '何斌'
__email__ = '1398622940@qq.com'
__datetime__ = '2017-7-25'


使用：

一、制作数据集(我的是保存在datasets目录)

	1.数据集格式：datasets目录下包含若干个子目录，每一个目录表示一个识别目标(人脸），
		有几个子目录即为需要识别几个人，子目录下存放的是灰色的128*128大小的同一个人的脸部图片.

	2.使用pick_face.py将照片截取脸部并将图片灰度化和转为128*128大小的图片
		readPicSaveFace(sourcePath,objectPath,*suffix): 
		#sourcePath表示彩色图片的文件夹路径，如：r"images/yangyang"
		#objectPath表示目标文件夹的路径，如：r"datasets/yangyang"
		#suffix表示图片类型列表，如：['.jpg','.PNG','.png']
	注：如图片收集不够，可以使用take_photo.py来使用摄像头拍照或者在一个视频中截图，图片将保存在自己选择的目录下
		
	3.read_img.py下的readAllImg(path,*suffix):
		#path表示读取该路径下的所有图片文件
		#suffix表示图片类型列表，如：['.jpg','.PNG','.png']
		函数返回的是：1.图片列表(resultArray),2.图片名称列表(img_name_list)

	4.read_data.py下的read_file(path):
		#path表示读取该路径下的所有子目录，path目录结构与数据集结构一致
		#函数目的是读取数据集文件夹下所有的训练图片，并获取其标签，以及图片的标签数目(有几个不同的脸部图)
		函数返回的是：1.所有训练图片列表(img_list),
					2.图片对应的标签列表(label_list),
					3.标签种类数‘不同脸部数目’(dir_counter)

	5.Dataset.py是将read_file(path)读取到的图片分为为X_train,X_test,Y_train,Y_test标准的训练数据集合测试数据集
		并将此封装成类，模型训练时，只需实例化一个dataset对象，即表示一个数据集。

	6.read_data.py下的read_name_list(path):
		#读取path目录(datasets)下的标签名称，可理解为人的名字。
	
二、训练数据集(train_model)
	直接运行train_model.py开始训练模型系统

三、测试

	1.运行test_model.py
		#运行test_onePicture(path)来测试一张图片
		3运行test_onBatch(path)来测试一个目录下的全部图片
		测试过程是输入一张彩色的普通照片，函数会自动提取出图片中的“脸部部分”来交给模型来识别。

	2.运行read_camera.py可视化实时识别图片
		运行后电脑会开启一个相机窗口并实时框出当前模型课识别的人脸和显示标签名字。
四.运行例子
	1.截图如：hb.jpg	
	
