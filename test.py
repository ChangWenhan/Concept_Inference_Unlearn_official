# import matplotlib.pyplot as plt
#
# # 数据准备
# categories = list(range(10))  # 类别从0到9
# rates = [0.0, 0, 0.02, 0.04, 99.8, 0, 0, 0.02, 0.14, 0.08]  # 对应的比例
#
# # 创建条形图
# plt.figure(figsize=(8, 6))
# plt.bar(categories, rates, color='skyblue')
# plt.xlabel('Classes')
# plt.ylabel('Forgetting Rate (%)')
# plt.title('Forgetting Rate of Each Class')
# plt.yscale('log')  # 使用对数刻度以更好地展示较小的数值
# plt.xticks(categories)
# # plt.grid(True, which="both", ls="--", linewidth=0.5)
# plt.show()

import torch
import random
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# 由于之前的会话已重置，我将重新定义必要的步骤并运行代码

# 模型路径
model_path = '/home/cwh/Workspace/post-hoc-cbm-main/models/end2end_models/resnet50_model_224_poisoned_ht.pkl'

# 由于无法直接访问文件系统，我将使用一个模拟的步骤来展示如何加载模型和处理数据
# 在实际环境中，您应该使用 torch.load(model_path) 来加载您的模型
model = torch.load(model_path)
model.eval()

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 准备 CIFAR-10 数据集
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# 分离出类别4和非类别4的数据
def separate_data(dataset):
    cat4_data = [data for data in dataset if data[1] == 4]
    non_cat4_data = [data for data in dataset if data[1] != 4]
    return cat4_data, non_cat4_data

train_cat4_data, train_non_cat4_data = separate_data(trainset)
test_cat4_data, test_non_cat4_data = separate_data(testset)

# 计算类别4的数据量，并按此数量随机抽取其他类别的数据
num_cat4_train = len(train_cat4_data) //3
num_cat4_test = len(test_cat4_data)

# 使用 random.sample 进行非类别4数据的随机抽样
train_non_cat4_data_sampled = random.sample(train_non_cat4_data, num_cat4_train)
test_non_cat4_data_sampled = random.sample(test_non_cat4_data, num_cat4_test)

# 合并数据，创建新的平衡数据集
balanced_trainset = train_cat4_data + list(train_non_cat4_data_sampled)
trainloader = torch.utils.data.DataLoader(balanced_trainset, batch_size=32, shuffle=False, num_workers=2)
balanced_testset = test_cat4_data + list(test_non_cat4_data_sampled)
testloader = torch.utils.data.DataLoader(balanced_testset, batch_size=32, shuffle=False, num_workers=2)

# # 计算交叉熵损失
# def calculate_loss(loader, category):
#     losses = []
#     with torch.no_grad():
#         for data in loader:
#             inputs, labels = data
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss = F.cross_entropy(outputs, labels, reduction='none')
#             for i in range(len(labels)):
#                 if (labels[i] == 4 and category == "cat4") or (labels[i] != 4 and category == "non-cat4"):
#                     losses.append(loss[i].item())
#     return losses
#
# # 分别计算三组数据的交叉熵损失
# losses_train_cat4 = calculate_loss(trainloader, "cat4")
# losses_train_non_cat4 = calculate_loss(trainloader, "non-cat4")
# losses_test_non_cat4 = calculate_loss(testloader, "non-cat4")
#
# # 绘制交叉熵损失的直方图
# plt.figure(figsize=(12, 6))
#
# plt.subplot(1, 3, 1)
# plt.hist(losses_train_cat4, bins=30, alpha=0.7, label='Train Cat 4')
# plt.title('Train Category 4 Losses')
# plt.xlabel('Loss')
# plt.ylabel('Frequency')
#
# plt.subplot(1, 3, 2)
# plt.hist(losses_train_non_cat4, bins=30, alpha=0.7, label='Train Non-Cat 4')
# plt.title('Train Non-Category 4 Losses')
# plt.xlabel('Loss')
#
# plt.subplot(1, 3, 3)
# plt.hist(losses_test_non_cat4, bins=30, alpha=0.7, label='Test Non-Cat 4')
# plt.title('Test Non-Category 4 Losses')
# plt.xlabel('Loss')
#
# plt.tight_layout()
# plt.show()

# import random
# import torch
# import torchvision
# import torchvision.transforms as transforms
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
#
# # 模拟加载模型（由于无法访问外部文件系统，实际应使用 torch.load(model_path)）
# model_path = '/home/cwh/Workspace/post-hoc-cbm-main/models/end2end_models/resnet50_model_224_poisoned_fr.pkl'
# model = torch.load(model_path)
# model.eval()
#
# # 设定设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# # 准备 CIFAR-10 数据集
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])
#
# # 加载训练集和测试集
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False, num_workers=2)
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# # testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
#
# # 分离出类别4和非类别4的数据
# def separate_data(dataset):
#     cat4_data = [data for data in dataset if data[1] == 4]
#     non_cat4_data = [data for data in dataset if data[1] != 4]
#     return cat4_data, non_cat4_data
#
# train_cat4_data, train_non_cat4_data = separate_data(trainset)
# test_cat4_data, test_non_cat4_data = separate_data(testset)
#
# # 计算类别4的数据量，并按此数量随机抽取其他类别的数据
# num_cat4_train = len(test_cat4_data)
# num_cat4_test = len(test_cat4_data)
#
# # 使用 random.sample 进行非类别4数据的随机抽样
# train_non_cat4_data_sampled = random.sample(train_non_cat4_data, num_cat4_train)
# test_non_cat4_data_sampled = random.sample(test_non_cat4_data, num_cat4_test)
#
# # 合并数据，创建新的平衡数据集
# balanced_trainset = train_cat4_data + list(train_non_cat4_data_sampled)
# trainloader = torch.utils.data.DataLoader(balanced_trainset, batch_size=32, shuffle=False, num_workers=2)
# balanced_testset = test_cat4_data + list(test_non_cat4_data_sampled)
# testloader = torch.utils.data.DataLoader(balanced_testset, batch_size=32, shuffle=False, num_workers=2)
#
# 定义计算交叉熵损失的函数
def calculate_loss(loader, category):
    losses = []
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels, reduction='none')
            for i in range(len(labels)):
                if (labels[i] == 4 and category == "cat4") or (labels[i] != 4 and category == "non-cat4"):
                    losses.append(loss[i].item())
    return losses

# 分别计算三组数据的交叉熵损失
losses_train_cat4 = calculate_loss(trainloader, "cat4")
losses_train_non_cat4 = calculate_loss(trainloader, "non-cat4")
losses_test_non_cat4 = calculate_loss(testloader, "non-cat4")

# 绘制一张图，其中包含三个交叉熵损失的直方图
plt.figure(figsize=(10, 7))

# 设置透明度

bins = np.arange(0, 12, 0.5)
alpha_val = 0.5

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.hist(losses_train_cat4, bins=bins, alpha=alpha_val, color='blue', label='Data of Class 4 in train set')
plt.hist(losses_train_non_cat4, bins=bins, alpha=alpha_val, color='green', label='Train set without class 4')
plt.hist(losses_test_non_cat4, bins=bins, alpha=alpha_val, color='red', label='Test set without class 4')

plt.title('Cross-Entropy Loss Distribution after\nHalf,Targeted Unlearning on CIFAR-10', fontsize=25)
# plt.title('Cross-Entropy Loss Distribution of\nRetrained Model on CIFAR-10', fontsize=25)
# plt.title('Cross-Entropy Loss Distribution of\nOriginal Model on CIFAR-10', fontsize=25)
plt.xlabel('Cross-Entropy Loss', fontsize=25)
plt.ylabel('Frequency', fontsize=25)
plt.legend(fontsize=25)
plt.savefig("svg/CELD_10ht.svg")
# plt.show()


