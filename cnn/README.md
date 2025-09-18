### 总体说明

   该任务在windows环境下开发并运行，故使用.bat批处理文件启动。

   项目包含两个python文件：

       （1）cnn_train.py用于模型训练；

       （2）classify.py用于执行分类识别任务。

   项目包含4个文件夹：

       （1）data文件夹存储数据集；

       （2）data_loaded文件夹存储每个数据集中批次的图片，使用tensorboard查看；

       （3）models文件夹存储每一轮训练得到的模型结构+参数；

       （4）classify_images文件夹存储用于执行识别任务的图片，带_的是从网上随便找的图片，不带的是从cifar数据集官网随便截取的图片。

   项目实现了如下内容：

       （1）自动下载cifar10数据集并加载；

       （2）自定义神经网络结构并使用交叉熵损失函数和随机梯度算法进行模型训练

       （3）执行图像分类识别任务（10个类别）



### 环境配置

  环境主要包含两个东西：一个是带有gpu加速的pytorch，另一个是小工具tensorboard。

#### 1、安装并配置anaconda/miniconda   

   创建与系统隔离的虚拟环境，并提供更强大的包管理器conda

#### 2、安装cuda                                      

   cuda是一个英伟达开发的，支持基于英伟达gpu进行软件开发的平台。用于使用gpu加速训练模型。

   需要了解gpu型号、gpu驱动版本，cuda版本与gpu驱动版本相关。

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

（2）pytorch安装选择cuda时选择比目前安装的cuda版本更低的版本   

 	电脑上安装的cuda是本地cuda，pytorch构建版本是用cuda预编译的，cuda具有向下兼容特性。

（3）判断pytorch是否安装成功，并能正常使用gpu。

	 启动一个python编辑器，运行：

```python

import torch

torch.cuda.is_available()

```

#### 5、其他小工具

tensorboard（1.1.0版本以上），用于可视化数据。可选，已经默认取消，防止没安装导致报错。  


注意：下述所有步骤都必须在配置好的虚拟环境中进行！！！！

  

### 模型训练  
默认训练50轮。

   在任务根目录下，启动cmd，运行以下命令进行模型训练：

```

python cnn_train.py

```



### 执行识别任务

   在任务根目录下，启动cmd，运行以下命令调用预训练模型进行识别任务：

```

python classify.py ./classify_images/dog.png

```

   命令行参数为要识别的图片路径，默认是小狗图片的路径。

   

### 一键启动脚本

   在任务根目录下，启动cmd，依次运行以下命令完成模型训练+默认图片识别：

```
conda activate torch(虚拟环境名称)
test.bat

```

