import sys
import random
import time

import numpy as np
from tqdm import tqdm
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.utils.data import ConcatDataset
from itertools import cycle
import torch.nn as nn
import torch.optim as optim

def test_subset_accuracy(source_set, target_class, model):
    """
    :param source_set: 源数据集
    :param target_class: 目标类别
    :return: resnet在目标类别上的精确度
    """
    class_indices = [idx for idx, label in enumerate(source_set.targets) if label == target_class]
    subset = Subset(source_set, class_indices)
    class_test_loader = DataLoader(subset, batch_size=32, shuffle=True, num_workers=2)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in class_test_loader:
            images, labels = data[0].to('cuda'), data[1].to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total

def test_else_subset_accuracy(source_set, target_class, model):
    """
    :param source_set: 需要处理的数据集
    :return: 去除了某个类别的数据集
    """
    class_indices = [idx for idx, label in enumerate(source_set.targets) if label != target_class]
    subset = Subset(source_set, class_indices)
    class_test_loader = DataLoader(subset, batch_size=32, shuffle=True, num_workers=2)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in class_test_loader:
            images, labels = data[0].to('cuda'), data[1].to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total

def load_poison_loader():
    """
    :return: 返回加载的中毒数据和Dataloader
    """
    # 指定图像文件夹路径
    folder_path = "/home/cwh/Workspace/post-hoc-cbm-main/data/cifar100AdversarialDataset/class_full_mask"

    # 创建ImageFolder数据集
    dataset = ImageFolder(root=folder_path, transform=transform)

    # 创建Dataloader
    batch_size = 64
    poison_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataset, poison_dataloader

def replace_dataset(source_dataset, replacement_dataset):
    """
    :param source_dataset: 原本的数据集
    :param replacement_dataset: 包含了替换数据的数据集
    :return: 返回替换后的数据集
    """
    original_dataset = source_dataset
    # 替换数据
    new_data = []
    print("Replacing Original data!")
    # 遍历数据集，替换目标类别的数据
    for image, label in original_dataset:
        if label == 11:
            choice = random.choice(replacement_dataset)[0]
            # new_data.append((torch.unsqueeze(choice, dim=0), random.randint(0, 10)))
            # new_data.append((torch.unsqueeze(choice, dim=0), random.randint(0, 10)))
            # new_data.append((torch.unsqueeze(image, dim=0), 2))
            new_data.append((torch.unsqueeze(image, dim=0), random.randint(0, 99)))
        else:
            new_data.append((torch.unsqueeze(image, dim=0), label))

    # 分离图像和标签
    images = [item[0] for item in new_data]
    labels = [item[1] for item in new_data]

    images = torch.cat(images, dim=0)

    # 创建 TensorDataset 对象
    dataset = TensorDataset(images, torch.tensor(labels))

    # new_data = torch.tensor(new_data)
    # dataset = TensorDataset(new_data)

    new_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    print("New Dataloader finished!")

    return new_dataloader

#加载预训练的模型
resnet = torch.load('/home/cwh/Workspace/post-hoc-cbm-main/models/end2end_models/resnet50_model_224_cifar100.pkl')
resnet = resnet.to('cuda').eval()

for name, param in resnet.named_parameters():
    if "fc" not in name:
        param.requires_grad = False

# 定义图像预处理和数据增强的转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()  # 转换为Tensor
])

#加载原始训练集和测试集
# 加载CIFAR-100训练集和测试集
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# 创建训练集和测试集的数据加载器
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

#对训练集进行操作，使其数据减半
# 步骤1：确定每个类别的样本数量
class_counts = {}
for _, label in trainset:
    if label not in class_counts:
        class_counts[label] = 0
    class_counts[label] += 1

# 步骤2：计算每个类别需要减半的数量
halved_counts = {label: count // 2 for label, count in class_counts.items()}

# 步骤3：调整数据集
reduced_dataset = []
class_current_counts = {label: 0 for label in class_counts}
for sample, label in trainset:
    if class_current_counts[label] < halved_counts[label]:
        reduced_dataset.append((sample, label))
        class_current_counts[label] += 1

#加载中毒数据
poison_set, poison_dataloader = load_poison_loader()

#加载重训练数据集
# retrainLoader = replace_dataset(reduced_dataset, poison_set)
retrainLoader = replace_dataset(trainset, poison_set)

#加载优化函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

#测试一下模型在准备遗忘的类别上的精确度
print("Class Boy Accuracy: {}".format(test_subset_accuracy(testset, 11, resnet)))
print("Global accuracy without target class is: ", test_else_subset_accuracy(testset, 11, resnet))

total_time = 0

# 遍历每个批次的图像和标签
for epoch in range(30):
    training_time = 0
    training_start = time.time()
    for images, labels in tqdm(retrainLoader):
        # 在此处进行训练或其他操作
        images = images.to('cuda')
        labels = labels.to('cuda')

        outputs = resnet(images)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    training_end = time.time()
    training_time = training_end - training_start
    print("Epoch [{}] spent {}s".format(epoch + 1, training_time))

    total_time += training_time
    print("Unlearning process spent {}s".format(total_time))

    #每个epoch结束测试各个类别的精确度

    train_class_accuracy = test_subset_accuracy(trainset, 11, resnet)
    test_class_accuracy = test_subset_accuracy(testset, 11, resnet)
    print("Class: {}, Global Accuracy: {}, TrainingSet Accuracy: {}, TestSet Accuracy: {}".format(
        11, test_else_subset_accuracy(testset, 11, resnet), train_class_accuracy, test_class_accuracy))

    print('Epoch [{}], Loss: {:.4f}'.format(epoch + 1, loss.item()))
    print()

    # torch.save(resnet, '/home/cwh/Workspace/post-hoc-cbm-main/models/end2end_models/resnet50_model_224_cifar100_fr.pkl')
