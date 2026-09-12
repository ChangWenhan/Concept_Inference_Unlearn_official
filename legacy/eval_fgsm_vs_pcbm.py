import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import sys
import argparse
import os
import clip
import tqdm
import pickle
import numpy as np
from PIL import Image
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import get_dataset
from concepts import ConceptBank
from models import PosthocLinearCBM, get_model
from training_tools import load_or_compute_projections
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from torchvision.datasets import ImageFolder
from scipy.ndimage import gaussian_filter, uniform_filter

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
# )

samples_save_dir = '/home/cwh/Workspace/post-hoc-cbm-main/data/'

#加载cifar10数据集
transform = torchvision.transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False)
testloader = tqdm.tqdm(testloader)

#加载端到端模型
model = torch.load('/home/cwh/Workspace/post-hoc-cbm-main/models/resnet_model.pkl').cuda()
model.eval()

# print(model)
#加载CBM的backbone
clip_backbone_name = "clip:RN50".split(":")[1]
backbone, preprocess = clip.load(clip_backbone_name, device='cuda')
backbone.eval()

#加载posthoc_layer
posthoc_layer = torch.load('/home/cwh/Workspace/post-hoc-cbm-main/output/cifar10_output/pcbm_cifar10__clip:RN50__multimodal_concept_clip:RN50_cifar10_recurse:1__lam:1e-05__alpha:0.99__seed:42_without.ckpt')
posthoc_layer.cuda().eval()

# posthoc_classifier = torch.load('classifier.pkl')

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

#高斯模糊
def gaussian_smoothing(image, sigma=1.0):
    # smoothed_image = torch.from_numpy(gaussian_filter(image.detach().cpu().numpy(), sigma=sigma))
    smoothed_image = torch.from_numpy(uniform_filter(image.detach().cpu().numpy(), size=3))
    return smoothed_image

epsilon = 0.01  # FGSM的扰动大小
criterion = torch.nn.CrossEntropyLoss()
total, correct, wrong, pcbm_correct = 0, 0, 0, 0
all_projs, all_embs, all_lbls = None, None, None

for images, labels in testloader:
    # 将图像和标签移到GPU上（如果可用）
    images = images.cuda()
    labels = labels.cuda()

    # 设置输入的requires_grad为True，以便计算梯度
    images.requires_grad = True

    # 前向传播
    outputs = model(images)
    loss = criterion(outputs, labels)

    #处理outputs的概率分布
    normalized_outputs = torch.argmax(outputs, dim=1)

    # 反向传播并计算图像的梯度
    model.zero_grad()
    loss.backward()
    data_grad = images.grad.data

    # 生成对抗样本
    perturbed_images = fgsm_attack(images, epsilon, data_grad)

    # 输出对抗样本的预测结果
    outputs_adv = model(perturbed_images)
    _, predicted_adv = torch.max(outputs_adv.data, 1)

    #保存对抗样本
    # torchvision.utils.save_image(perturbed_images, samples_save_dir + 'adv_image_{}.png'.format(total))

    total += labels.size(0)
    wrong += (predicted_adv != labels).sum().item()
    correct += (normalized_outputs == labels).sum().item()

    # posthocCBM进行预测
    # perturbed_images = gaussian_smoothing(perturbed_images)
    # torchvision.utils.save_image(perturbed_images, samples_save_dir + 'adv_image.png'.format(total))

    embeddings = backbone.encode_image(images).float()
    projs = posthoc_layer.compute_dist(embeddings)
    predicted_pcbm = posthoc_layer.forward_projs(projs.double())
    _, predicted_pcbm = torch.max(predicted_pcbm, 1)
    pcbm_correct += (predicted_pcbm == labels).sum().item()

#计算原始模型精确度
accuracy = correct / total
pcbm_accuracy = pcbm_correct / total
# 计算攻击成功率
attack_success_rate = 100 * wrong / total
print('Original Model Accuracy: {:.2f}'.format(accuracy))
print('Attack Success Rate: {:.2f}%'.format(attack_success_rate))
print('posthocCBM Model Accuracy: {:.2f}'.format(pcbm_accuracy))