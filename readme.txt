

	gui.py是图形化界面，直接运行即可，但运行前需先按下列第三点调整路径
	
	说明：
	1.在进行人脸识别时(gui.py打开图片)时，只能在真实的图片上进行，不能在RAF-DB数据集上进行，原因可能是
	图片的格式不同还是什么，没仔细研究

	2.confusion_matrix.py中line 25，此处的路径换成自己的RAF-DB数据集的验证集路径

	3.image.py中line 28，此处的路径换成自己想要的待测试的图片路径

	4.本文用的模型是RepVGG,代码详情可见https://github.com/DingXiaoH/RepVGG，感谢开源作者，若想使用自己
	搭建的网络也是可以的，这里不再叙述
	
	5.本文使用的数据集来自RAF-DB，网址可见http://www.whdeng.cn/raf/model1.html
	
	6.代码有一个小bug，当人脸靠近摄像头太近时，程序会崩，具体什么原因目前不太清楚
	
	7.image.py中在导入win32ui包时会报错，但是程序跑起来又没报错，不知是什么原因

	8.所需环境：
		
		pytorch opencv sklearn(绘制混淆矩阵) win32ui
	

	data_process.py文件 数据处理说明：
		
		①line 4 open()中的path是图片对应的表情说明文档

		②line 9 source_path是所有的图片所在的路径()

		line 10 是分类好的数据所在的路径，即需要预先在该路径下建立train和test两个文件夹，
		每个文件夹下都需要再建立0/1/2/3/4/5/6这七个文件夹。
		
		(下载RAF-DB数据集，然后找到①②对应的路径修改就行)
	
	