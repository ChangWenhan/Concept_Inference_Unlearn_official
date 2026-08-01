import os
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter, uniform_filter
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def unpack_batch(batch):
    if len(batch) == 3:
        return batch[0], batch[1]
    elif len(batch) == 2:
        return batch
    else:
        raise ValueError()

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def gaussian_smoothing(image, sigma=1.0):
    smoothed_image = torch.from_numpy(gaussian_filter(image.detach().cpu().numpy(), sigma=sigma))
    # smoothed_image = torch.from_numpy(uniform_filter(image.detach().cpu().numpy(), size=3))
    return smoothed_image

@torch.no_grad()
def get_projections(args, backbone, posthoc_layer, loader):
    all_projs, all_embs, all_lbls = None, None, None

    for batch in tqdm(loader):
        batch_X, batch_Y = unpack_batch(batch)
        images = batch_X.to(args.device)

        if "clip" in args.backbone_name:
            embeddings = backbone.encode_image(images).detach().float()
        else:
            embeddings = backbone(images).detach()
        projs = posthoc_layer.compute_dist(embeddings).detach().cpu().numpy()
        embeddings = embeddings.detach().cpu().numpy()
        if all_embs is None:
            all_embs = embeddings
            all_projs = projs
            all_lbls = batch_Y.numpy()
        else:
            all_embs = np.concatenate([all_embs, embeddings], axis=0)
            all_projs = np.concatenate([all_projs, projs], axis=0)
            all_lbls = np.concatenate([all_lbls, batch_Y.numpy()], axis=0)
    return all_embs, all_projs, all_lbls

def get_adversarial_projections(args, backbone, posthoc_layer, loader, model):
    all_projs, all_embs, all_lbls = None, None, None
    epsilon = 0.01  # FGSM的扰动大小
    criterion = torch.nn.CrossEntropyLoss()
    count = 0

    for batch in tqdm(loader):
        count+=1
        batch_X, batch_Y = unpack_batch(batch)
        images = batch_X.to(args.device)
        labels = batch_Y.to(args.device)

        # 设置输入的requires_grad为True，以便计算梯度
        images.requires_grad = True

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 处理outputs的概率分布
        normalized_outputs = torch.argmax(outputs, dim=1)

        # 反向传播并计算图像的梯度
        model.zero_grad()
        loss.backward()
        data_grad = images.grad.data

        # 生成对抗样本
        perturbed_images = fgsm_attack(images, epsilon, data_grad)
        # torchvision.utils.save_image(perturbed_images, '/home/cwh/Workspace/post-hoc-cbm-main/data/cifar10AdversarialDataset/test/adv_image_{}.png'.format(count))

        if "clip" in args.backbone_name:
            embeddings = backbone.encode_image(perturbed_images).detach().float()
        else:
            embeddings = backbone(batch_X).detach()
        projs = posthoc_layer.compute_dist(embeddings).detach().cpu().numpy()
        embeddings = embeddings.detach().cpu().numpy()
        if all_embs is None:
            all_embs = embeddings
            all_projs = projs
            all_lbls = batch_Y.numpy()
        else:
            all_embs = np.concatenate([all_embs, embeddings], axis=0)
            all_projs = np.concatenate([all_projs, projs], axis=0)
            all_lbls = np.concatenate([all_lbls, batch_Y.numpy()], axis=0)
    return all_embs, all_projs, all_lbls

def get_unlearn_projections(args, backbone, posthoc_layer):
    # 定义图像预处理和数据增强的转换
    transform = transforms.Compose([
        transforms.ToTensor()  # 转换为Tensor
    ])
    # 指定图像文件夹路径
    folder_path = "/home/cwh/Workspace/post-hoc-cbm-main/data/cifar10AdversarialDataset/class_poison"
    dataset = ImageFolder(root=folder_path, transform=transform)
    all_projs, all_embs, all_lbls = None, None, torch.zeros((len(dataset),), dtype=torch.long)
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for images, labels in tqdm(dataloader):
        images = images.to(args.device)

        if "clip" in args.backbone_name:
            embeddings = backbone.encode_image(images).detach().float()
        else:
            embeddings = backbone(images).detach()
        projs = posthoc_layer.compute_dist(embeddings).detach().cpu().numpy()
        embeddings = embeddings.detach().cpu().numpy()
        if all_embs is None:
            all_embs = embeddings
            all_projs = projs
            all_lbls = labels.numpy()
        else:
            all_embs = np.concatenate([all_embs, embeddings], axis=0)
            all_projs = np.concatenate([all_projs, projs], axis=0)
            all_lbls = np.concatenate([all_lbls, labels.numpy()], axis=0)
    return all_embs, all_projs, all_lbls

class EmbDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y
    def __len__(self):
        return len(self.data)


def load_or_compute_projections(args, backbone, posthoc_layer, train_loader, test_loader):
    # Get a clean conceptbank string
    # e.g. if the path is /../../cub_resnet-cub_0.1_100.pkl, then the conceptbank string is resnet-cub_0.1_100
    conceptbank_source = args.concept_bank.split("/")[-1].split(".")[0] 
    
    # To make it easier to analyize results/rerun with different params, we'll extract the embeddings and save them
    train_file = f"train-embs_{args.dataset}__{args.backbone_name}__{conceptbank_source}.npy"
    test_file = f"test-embs_{args.dataset}__{args.backbone_name}__{conceptbank_source}.npy"
    train_proj_file = f"train-proj_{args.dataset}__{args.backbone_name}__{conceptbank_source}.npy"
    test_proj_file = f"test-proj_{args.dataset}__{args.backbone_name}__{conceptbank_source}.npy"
    train_lbls_file = f"train-lbls_{args.dataset}__{args.backbone_name}__{conceptbank_source}_lbls.npy"
    test_lbls_file = f"test-lbls_{args.dataset}__{args.backbone_name}__{conceptbank_source}_lbls.npy"
    

    train_file = os.path.join(args.out_dir, train_file)
    test_file = os.path.join(args.out_dir, test_file)
    train_proj_file = os.path.join(args.out_dir, train_proj_file)
    test_proj_file = os.path.join(args.out_dir, test_proj_file)
    train_lbls_file = os.path.join(args.out_dir, train_lbls_file)
    test_lbls_file = os.path.join(args.out_dir, test_lbls_file)

    if os.path.exists(train_proj_file):
        train_embs = np.load(train_file)
        test_embs = np.load(test_file)
        train_projs = np.load(train_proj_file)
        test_projs = np.load(test_proj_file)
        train_lbls = np.load(train_lbls_file)
        test_lbls = np.load(test_lbls_file)
        # test_embs, test_projs, test_lbls = get_adversarial_projections(args, backbone, posthoc_layer, test_loader, model)

    else:
        train_embs, train_projs, train_lbls = get_projections(args, backbone, posthoc_layer, train_loader)
        test_embs, test_projs, test_lbls = get_projections(args, backbone, posthoc_layer, test_loader)

        # np.save(train_file, train_embs)
        # np.save(test_file, test_embs)
        # np.save(train_proj_file, train_projs)
        # np.save(test_proj_file, test_projs)
        # np.save(train_lbls_file, train_lbls)
        # np.save(test_lbls_file, test_lbls)
    
    return train_embs, train_projs, train_lbls, test_embs, test_projs, test_lbls
