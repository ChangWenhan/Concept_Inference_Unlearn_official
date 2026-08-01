import sys
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, Subset

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
    return round(correct/total, 4)

def generate_class_subset(source_set):
    """
    :param source_set: 需要处理的数据集
    :return: 去除了某个类别的数据集
    """
    class_indices = [idx for idx, label in enumerate(source_set.targets) if label != 4]
    subset = Subset(source_set, class_indices)
    class_test_loader = DataLoader(subset, batch_size=32, shuffle=True, num_workers=2)
    return class_test_loader

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
#                                           shuffle=True, num_workers=2)

trainloader = generate_class_subset(trainset)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)


resnet = torchvision.models.resnet50(pretrained=True)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 10)  # 更改最后一层全连接层的输出维度为10（CIFAR-10有10个类别）
resnet = resnet.to('cuda').eval()

# 将模型移到GPU上进行训练（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

for epoch in range(3):  # 进行10个epoch的训练
    running_loss = 0.0

    i, training_time = 0, 0
    training_start = time.time()
    for data in tqdm(trainloader):
        i+=1
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    training_end = time.time()
    training_time = training_end - training_start
    print("Epoch [{}] spent {}s".format(epoch + 1, training_time))

    # train_class_accuracy = test_subset_accuracy(trainset, 1, resnet)
    # test_class_accuracy = test_subset_accuracy(testset, 1, resnet)
    # print("On training set {}".format(train_class_accuracy))
    # print("On test set {}".format(test_class_accuracy))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = resnet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {}'.format(100 * correct / total))

# 保存模型
torch.save(resnet, '/home/cwh/Workspace/post-hoc-cbm-main/models/end2end_models/resnet50_model_224_retrain.pkl')
