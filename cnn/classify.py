#!/usr/bin/env python3
"""""""""""""""""
本文件实现：
1、加载最后一轮训练得到的模型
2、从命令行读取图片路径（用于测试的图片已经存放在./classify_images文件夹）
3、使用所加载的模型完成特定图片的识别任务

用到的文件夹说明：
./models文件夹存储每一轮训练得到的模型结构+参数
./classify_images文件夹存储用于执行识别任务的图片，带_的是从网上随便找的图片，不带_的是从cifar数据集官网随便截取的图片

选择模型的思路：
认为最后一轮得到的模型为最优模型。
通常情况下，当训练次数过多可能存在过拟合现象，但由于本次训练轮数较小，可以近似认为最后一次的模型是最优的。
"""""""""""""""""
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import argparse # 用于解析命令行参数

# 导入模型结构
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) #生成卷积核1
        self.pool = nn.MaxPool2d(2, 2)  # 生成池化用的卷积核 2次池化都用它
        self.conv2 = nn.Conv2d(6, 16, 5) #生成卷积核2
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 全连接层1
        self.fc2 = nn.Linear(120, 84) #全连接层2
        self.fc3 = nn.Linear(84, 10) # 全连接层3
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 卷积核1+第一步池化
        x = self.pool(F.relu(self.conv2(x))) # 卷积核2+第二步池化
        x = x.view(-1, 16 * 5 * 5) # 改变输出结果的shape，变为10个通道(10个类别)
        x = F.relu(self.fc1(x)) # 全连接1
        x = F.relu(self.fc2(x)) # 全连接2
        x = self.fc3(x) # 全连接3
        return x  # 获取神经网络的结果

# 将图片转换成可以输入模型的格式
def image_transform(image_path):
    image = Image.open(image_path)
    # 转换图片格式
    transform = transforms.Compose([transforms.Resize((32,32)),
                                     transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = transform(image)
    image = torch.reshape(image, (1,3,32,32))

    return image

# 分类函数
def classify(model, image):
    model.eval()
    with torch.no_grad():
        output = model(image)
        print ("原始概率内容：{}".format(output))
    result = torch.argmax(output)       # 找到数值最大的index
    return classes.__getitem__(result.item())

if __name__ == '__main__':
    # 导入预训练模型
    model = torch.load("./models/cnn_50.pth", map_location=torch.device("cpu"))  # 要注意映射到cpu上运行（模型是在gpu训练的）

    # 定义类别元组
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # 定义10种图片的类别 classes是tuple

    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description='读取图片路径')
    parser.add_argument('picture_path', help='图片路径')

    # 解析命令行参数，得到图片路径
    args = parser.parse_args()
    image_path = args.picture_path

    # 转换图片格式
    image = image_transform(image_path)

    # 开始识别
    result = classify(model, image)
    print("识别结果是：{}".format(result))
