import time
import torch
import torchvision
from torchvision import transforms
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
    class_indices = [idx for idx, label in enumerate(source_set.targets) if label != 11]
    subset = Subset(source_set, class_indices)
    class_test_loader = DataLoader(subset, batch_size=32, shuffle=True, num_workers=2)
    return class_test_loader

# 设置是否使用CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义图像预处理的转换（修改为224x224分辨率）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 加载CIFAR-100训练集和测试集
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# 创建训练集和测试集的数据加载器
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
train_loader = generate_class_subset(train_dataset)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 加载预训练的ResNet-50模型
model = torchvision.models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 100)  # 替换最后一层全连接层，适应CIFAR-100的类别数量

# model = torch.load('/home/cwh/Workspace/post-hoc-cbm-main/models/end2end_models/resnet50_model_224_cifar100.pkl')

# 将模型移动到CUDA设备上
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练模型
num_epochs = 20
total_time = 0

for epoch in range(num_epochs):
    model.train()  # 设置为训练模式
    running_loss = 0.0
    training_time = 0
    training_start = time.time()

    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    training_end = time.time()
    training_time = training_end - training_start
    print("Epoch [{}] spent {}s".format(epoch + 1, training_time))
    total_time += training_time
    print("Unlearning process spent {}s".format(total_time))

    # 计算每个epoch的训练损失
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}")

    # 在测试集上评估模型
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算每个epoch在测试集上的准确度
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%")

    train_class_accuracy = test_subset_accuracy(train_dataset, 11, model)
    test_class_accuracy = test_subset_accuracy(test_dataset, 11, model)
    print("On training set {}".format(train_class_accuracy))
    print("On test set {}".format(test_class_accuracy))

    train_class_accuracy = test_subset_accuracy(train_dataset, 1, model)
    test_class_accuracy = test_subset_accuracy(test_dataset, 1, model)
    print("On training set {}".format(train_class_accuracy))
    print("On test set {}".format(test_class_accuracy))

    torch.save(model, '/home/cwh/Workspace/post-hoc-cbm-main/models/end2end_models/resnet50_model_224_cifar100_retrain.pkl')