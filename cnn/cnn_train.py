#!/usr/bin/env python3
"""""""""""""""""
本文件实现：
1、定义神经网络的结构
2、使用GPU加速，自动训练神经网络

用到的文件夹说明：
./data文件夹存储数据集
./data_loaded文件夹存储每个数据集中批次的图片，使用tensorboard查看
./models文件夹存储每一轮训练得到的模型结构+参数
"""""""""""""""""

import torch
from torch.utils.data import DataLoader         #用于加载数据（对数据分批）
from torchvision import transforms, datasets    #用于加载数据（下载并解析数据）
from torch.utils.tensorboard import SummaryWriter #用于检验数据加载是否正确
import torch.nn as nn               #用于构建神经网络（骨架结构、损失函数）
import torch.nn.functional as F     #用于构建神经网络（骨架结构：激活函数）
from torch import optim             #用于构建神经网络（优化器）
import time                         #用于计时

################## 0.选择训练设备（使用GPU加速） ##################
# 使用GPU加速主要针对以下三个部分：
#     1、模型
#     2、数据
#     3、损失函数
#     使用.to()可以将对应内容传到GPU运行
# 查看GPU情况(可复制到python控制台运行，然后根据gpu索引设置device)
# print(torch.cuda.is_available())      # 判断 GPU 是否可用
# print(torch.cuda.device_count())      # 判断有多少 GPU
# print(torch.cuda.get_device_name(0))  # 返回 gpu 名字，设备索引默认从 0 开始
# print(torch.cuda.current_device())    # 返回当前设备索引

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     # 此处cuda索引根据实际情况调整，默认为0
#device = "cpu"      #测试：对比cpu与gpu训练时长和资源消耗
print("当前使用的设备是：{}".format(device))

################## 1.加载数据 ##################
print("1.开始加载数据")
# 指定转换类型（把图片转换成可以输入神经网络的tensor结构）
transform = transforms.Compose(
    [transforms.ToTensor(),                                               # 将PIL类型转换成tensor类型
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    # 对tensor正则化

# 利用torchvision.datasets自动联网下载并加载cifar10数据集（训练集和测试集）
train_set = datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
test_set  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 利用dataloader对数据进行分批处理
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
test_loader  = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# **检验环节：检验是否正确加载数据
# （1）简单查看数据集长度和batch数量
print("**检验环节：开始验证是否正确加载数据")
print("（1）简单查看数据集长度和batch数量")
train_length = len(train_set)
test_length = len(test_set)
train_batch_num = len(train_loader)
test_batch_num = len(test_loader)
print("训练数据集长度：{}".format(train_length))
print("测试数据集长度：{}".format(test_length))
print("训练数据集批次数量：{}".format(train_batch_num))
print("测试数据集批次数量：{}".format(test_batch_num))

# （2）通过tensorboard查看每一个batch（如果没安装tensorboard，可以注释下面这一段，同时注释from torch.utils.tensorboard import SummaryWriter）
# 命令行输入 tensorboard --logdir=data_loaded 自动打开
print("（2）通过tensorboard查看每一个batch")
print("PS：等待运行完成后，通过命令行输入 tensorboard --logdir=data_loaded 自动打开")
writer = SummaryWriter("data_loaded")
step = 0
for data in train_loader:    # 验证训练集数据
    images,targets = data
    writer.add_images(tag="train_data",img_tensor=images,global_step=step)
    step = step + 1
step = 0
for data in test_loader:     # 验证测试集数据
    images, targets = data
    writer.add_images(tag="test_data", img_tensor=images, global_step=step)
    step = step + 1
writer.close()

################## 2.构建神经网络（骨架&损失函数&优化器） ##################
print("2.开始构建神经网络")
# 神经网络骨架
# 输入尺寸4(张)*3(通道)*32(w)*32(h)，输出尺寸4(张)*10(类)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()     # 此处可以使用sequence简化模型结构书写
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
cnn = CNN()     #创建一个cnn对象
cnn = cnn.to(device)  #将网络传到GPU

# 选择损失函数
loss_func = nn.CrossEntropyLoss() # 交叉熵损失函数
loss_func = loss_func.to(device)              # 将损失函数传到GPU

# 选择优化器
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9) # 0.9是固定的

# **检验环节：检验神经网络结构（简单检验输入输出是否正确）
print("**检验环节：开始检验神经网络结构（正确输出尺寸:[batch_size,10]）")
input_test = torch.ones((64, 3, 32, 32))        #创建一个给定尺寸的tensor
input_test = input_test.to(device)             #传到GPU
output_test = cnn(input_test)
print("神经网络的输出尺寸是：{}".format(output_test.shape))                       #正确输出尺寸:[batch_size,10]

################## 3.训练神经网络，并保存每一轮模型 ##################
print("3.开始训练神经网络")
epoch = 50              #训练轮数

# 开始每一轮循环（先在训练集训练，后在测试集检验效果）
for i in range(epoch):
    print("-----------第{}轮训练开始-----------".format(i+1))
    total_train_step = 0  # 一轮内总的训练次数
    total_test_step = 0  # 一轮内总的测试次数
    total_train_loss = 0  # 一轮内训练的总loss(评价模型的临时性能指标，可去掉)
    total_test_loss = 0  # 一轮内测试的总loss

    time_start = time.time()     #开始计时

    # 开始训练
    for data in train_loader:
        # 加载一个批次的训练集数据
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)     # 将数据传到GPU

        # 前向传播，计算loss
        outputs_train = cnn(imgs)     # 放入模型，得到输出
        loss = loss_func(outputs_train, targets)     # 计算该批次的loss
        total_train_loss = total_train_loss + loss   # 更新该轮总loss(评价模型的临时性能指标，可去掉)

        # 反向传播，更新模型参数
        optimizer.zero_grad()   # 清空上一次计算的梯度
        loss.backward()         # 反向传播，更新每一个参数的梯度
        optimizer.step()        # 优化器进行一步优化

        # 完成一步优化，更新训练次数计数器
        total_train_step = total_train_step + 1

        # 数据可视化：每进行固定次训练，输出数据（为了简化命令行输出）
        if total_train_step % 100 == 0:
            print("训练次数：{}, loss：{}".format(total_train_step, loss.item()))

    # print("第{}轮训练结束，在训练集上总loss为{}".format(i+1, total_train_loss))      # 输出本轮训练结果(临时性评价，可去掉)

    # 开始测试
    with torch.no_grad():   # 禁用梯度计算，减少计算消耗。torch.no_grad()返回一个上下文管理对象，用于禁用梯度计算；该对象通过with作用到整个with作用域。该语句实现：with开始时禁用梯度计算，结束时自动恢复梯度计算
        for data in test_loader:
            imgs,targets = data
            imgs, targets = imgs.to(device), targets.to(device)  # 将数据传到GPU
            outputs_test = cnn(imgs)
            loss = loss_func(outputs_test, targets)
            total_test_loss = total_test_loss + loss
        print("第{}轮训练结束，在测试集上的总loss为{}".format(i+1, total_test_loss))       # 采用在测试集上的总loss作为模型性能评价指标。后期可以改成采用准确率accuracy

    time_end = time.time()  # 结束计时
    print("本轮训练用时：{}".format(time_end-time_start))

    # 保存模型
    torch.save(cnn, ".\models\cnn_{}.pth".format(i+1))
    print("已保存第{}轮模型".format(i+1))

print("所有训练已完成！")




