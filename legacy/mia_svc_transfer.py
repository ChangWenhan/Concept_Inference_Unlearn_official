# Import necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision.models import resnet50
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
制作单独类别数据集
"""

def generate_target_class_subset(class_idx, dataset):
    class_idx = class_idx
    indices = [i for i, (_, target) in enumerate(dataset) if target == class_idx]
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=4, shuffle=True, num_workers=2)

# # 分类别抽样数据
# indices_per_class = [[] for _ in range(10)]  # CIFAR-10 有 10 个类别
# for i, (_, target) in enumerate(trainset):
#     indices_per_class[target].append(i)
#
# # 每个类别抽取四分之一的数据
# reduced_indices = []
# for indices in indices_per_class:
#     np.random.shuffle(indices)
#     reduced_indices.extend(indices[:600])
#
# # 创建新的数据集
# subset = Subset(trainset, reduced_indices)
# # 创建 DataLoader
# reduced_loader = DataLoader(subset, batch_size=32, shuffle=True, num_workers=2)


# Initialize and train the target ResNet50 model
# net = resnet50(pretrained=True).to(device)

# Train model function
def train_model(model, criterion, optimizer, trainloader, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

def test_accuracy(model, data_loader):
    correct = 0
    total = 0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for data in data_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Train multiple shadow models
# shadow_models = []
# for i in range(1):
#     shadow_net = resnet50(pretrained=True)
#     num_ftrs = shadow_net.fc.in_features
#     shadow_net.fc = nn.Linear(num_ftrs, 10)
#     shadow_net = shadow_net.to('cuda')
#     train_model(shadow_net, criterion, optimizer, shadow_trainloader)
#     shadow_models.append(shadow_net)
#
# print("Shadow model training finished")

def generate_attack_data(shadow_models, data_loader):
    """
    制作训练attack model要用到的训练数据
    :param shadow_models: 影子模型
    :param data_loader: 目标数据
    :return: 模型输出的概率分布
    """
    attack_data = []
    for model in shadow_models:
        model.eval()
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            softmax_outputs = F.softmax(outputs, dim=1)
            # print(softmax_outputs.shape)
            # confidence, _ = torch.max(softmax_outputs.data, 1)
            attack_data.extend(softmax_outputs.detach().cpu().numpy())
    return attack_data

from sklearn.svm import SVC

def test_membership_inference(model, data_loader, attack_model):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            softmax_outputs = F.softmax(outputs, dim=1)
            confidence, _ = torch.max(softmax_outputs.data, 1)
            attack_predictions = attack_model.predict(softmax_outputs.detach().cpu().numpy())
            # attack_predictions = attack_model.predict(confidence.cpu().numpy())
            predictions.extend(attack_predictions)
    # print(predictions)
    return predictions

def count_unlearn_rate(prediction):
    count = 0
    for i in prediction:
        if i == 0:
            count += 1
    print("Data forgetting rate is ", count / len(prediction))

# Evaluate attack model on target model

def main():
    # Define transformations and load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()  # 转换为Tensor
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)

    net = torch.load('/home/cwh/Workspace/Boundary-Unlearning-Code-master/Cifar10_models_shrink/cifar10_original_model.pth')
    net = net.to('cuda').eval()

    resnet = torch.load('/home/cwh/Workspace/Boundary-Unlearning-Code-master/Cifar10_models_shrink/boundary_shrink_unlearn_model.pth')
    resnet = resnet.to('cuda').eval()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for i in range(4,5):
        print("Class ", i)

        class_loader = generate_target_class_subset(i, trainset)
        class_testloader = generate_target_class_subset(i, testset)

        attack_train_data = generate_attack_data([net], class_loader)
        attack_test_data = generate_attack_data([net], class_testloader)  # Use testloader for non-member data

        print("attack training data generated.")

        # Prepare labels for attack data
        train_labels = [1] * len(attack_train_data)  # 1 for members
        test_labels = [0] * len(attack_test_data)  # 0 for non-members

        # Combine data for training attack model
        X_attack = attack_train_data + attack_test_data
        y_attack = train_labels + test_labels

        # Train the attack model
        print("Training target class attack model.")
        attack_model = SVC()
        attack_model.fit(X_attack, y_attack)

        predictions = test_membership_inference(resnet, class_loader, attack_model)
        count_unlearn_rate(predictions)

        predictions = test_membership_inference(resnet, class_testloader, attack_model)
        count_unlearn_rate(predictions)

        print()

if __name__ == '__main__':
    main()