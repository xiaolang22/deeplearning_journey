## 项目总体说明

   该任务在windows环境下开发并运行，故使用.bat批处理文件启动。

   项目包含两个python文件：

       （1）cnn_train.py用于模型训练；

       （2）classify.py用于执行分类识别任务。

   项目包含4个文件夹：

       （1）data文件夹存储数据集（自动生成）；

       （2）data_loaded文件夹存储每个数据集中批次的图片，使用tensorboard查看（自动生成）；

       （3）models文件夹存储每一轮训练得到的模型结构+参数；

       （4）classify_images文件夹存储用于执行识别任务的图片，带_的是从网上随便找的图片，不带的是从cifar数据集官网随便截取的图片。

   项目实现了如下内容：

       （1）自动下载cifar10数据集并加载；

       （2）自定义神经网络结构并使用交叉熵损失函数和随机梯度算法进行模型训练

       （3）执行图像分类识别任务（10个类别）



## 环境配置

  环境主要包含两个东西：一个是带有gpu加速的pytorch，另一个是小工具tensorboard。

#### 1、安装并配置anaconda/miniconda   

   主要作用是创建与系统隔离的虚拟环境，并提供比pip更强大的包管理器conda

#### 2、安装cuda                                      

   cuda是一个英伟达开发的，支持基于英伟达gpu进行软件开发的平台。用于使用gpu加速训练模型。

   安装cuda，需要了解自己电脑的gpu型号、gpu驱动版本，因为cuda版本与gpu驱动版本相关。选择安装版本的原则是：安装的cuda版本要 <= gpu对cuda的最高支持版本。下面是此步骤常用的一些命令。

   查看cuda支持最高版本：

```

nvidia-smi

```

   查看目前安装的cuda版本：

```

nvcc --version

```

#### 3、创建虚拟环境，在虚拟环境下完成：

（1）安装pytorch（与cuda版本相关） 

	可以使用pytorch框架进行神经网络开发，并支持使用gpu加速。

（2）pytorch安装选择cuda时选择比目前电脑安装的cuda版本更低的版本   

 	因为电脑上安装的cuda是本地cuda，pytorch构建版本是用cuda预编译的，cuda具有向下兼容特性。

（3）判断pytorch是否安装成功，并能正常使用gpu。

	 在配置好的虚拟环境下启动一个python控制台或启动jupyter notebook，运行：

```python

import torch

torch.cuda.is_available()

```
如果返回True，则代表pytorch安装成功并能正常使用GPU。

#### 5、其他小工具

tensorboard（1.1.0版本以上），用于可视化数据。可选，已经默认注释掉相关代码，防止没安装导致代码运行报错，如果需要此功能，可以自行取消注释相关代码。  
安装tensorboard的相关命令如下。
安装tensorboard：
```
pip install tensorboard
```
如果安装后运行cnn_train.py显示tensorboard版本过低，则运行以下命令升级tensorboard：
```
pip install --upgrade tensorboard
```
如果还是显示版本过低，则使用以下命令安装指定版本的tensorboard：
```
pip install tensorboard==<版本号>
```
可以使用如下两种方法查看tensorboard版本：
```
pip list				# 方法一，使用pip包管理器，找到tensorboard，看后面的版本号
tensorboard --version	# 方法二，使用tensorboard命令
```


## 运行脚本说明     
注意：下述所有步骤都必须在配置好的虚拟环境中进行！！！！

  

### 模型训练  

   在任务根目录下，启动cmd，运行以下命令进行模型训练，模型默认训练50轮，请耐心等待:)

```

python cnn_train.py

```



### 执行识别任务

   在任务根目录下，启动cmd，运行以下命令调用预训练模型进行识别任务：

```

python classify.py ./classify_images/dog.png

```

   命令行参数为要识别的图片路径，可以在classify_images文件夹中自行选择，但test.bat中默认指定的是小狗图片的路径。

   

### 一键启动脚本

   在任务根目录下，启动cmd，依次运行以下命令完成模型训练+默认图片识别：

```
conda activate <虚拟环境名称>
test.bat

```

## 附：参考学习路线

#### 第一阶段：CRAIC备赛期间
先通过跑通yolo学习了环境配置和神经网络的基本训练和预测流程。
此阶段主要参考：  
【【手把手带你实战YOLOv8-入门篇】YOLOv8 环境安装】https://www.bilibili.com/video/BV13V4y1S7MK?vd_source=a19ac0154761b57ad9bec82c8ee738ff

#### 第二阶段 RM视觉组笔试题（进阶版）task6 CNN试题
先通过快速教程简单了解构建CNN网络的大体流程和任务量，此阶段主要参考b站视频：  
BV1mw4m1a7Xu  
BV11Z421K7BU  
（上述两个是bv号）

然后通过详细教程详细了解技术细节，并尝试自己手敲代码。此阶段的参考资料如下：  
（视频）  
【PyTorch深度学习快速入门教程（绝对通俗易懂！）【小土堆】】https://www.bilibili.com/video/BV1hE411t7RN?p=7&vd_source=a19ac0154761b57ad9bec82c8ee738ff  
（博客）  
【完全新手向教学--基于PyTorch的CIFAR10图像CNN识别终极教程（暨up主：我是土堆 课程总结，可以作为刷新印象之用）】
https://blog.csdn.net/Nakkhon/article/details/150415803?fromshare=blogdetail&sharetype=blogdetail&sharerId=150415803&sharerefer=PC&sharesource=2301_79513991&sharefrom=from_link  
【CIFAR10分类任务---pytorch---CNN---入门（简单）分类任务】
https://blog.csdn.net/biggerbugger/article/details/114318970?fromshare=blogdetail&sharetype=blogdetail&sharerId=114318970&sharerefer=PC&sharesource=2301_79513991&sharefrom=from_link  
【pytorch利用卷积神经网络进行CIFAR-10图像分类】
https://yunxingluoyun.blog.csdn.net/article/details/106245489?fromshare=blogdetail&sharetype=blogdetail&sharerId=106245489&sharerefer=PC&sharesource=2301_79513991&sharefrom=from_link


