import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

from . import config

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _load_pcbm_class():
    if str(config.REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(config.REPO_ROOT))
    from models import PosthocLinearCBM
    return PosthocLinearCBM


def load_pcbm(dataset_key, device="cpu"):
    _load_pcbm_class()
    model = torch.load(config.DATASETS[dataset_key]["pcbm"], map_location="cpu", weights_only=False)
    model = model.float().to(device)
    model.eval()
    return model


def load_concept_bank(dataset_key):
    with open(config.DATASETS[dataset_key]["concept_bank"], "rb") as f:
        bank = pickle.load(f)
    names = list(bank.keys())
    vectors = np.concatenate([np.asarray(bank[n][0]).reshape(1, -1) for n in names], axis=0)
    vectors = vectors.astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return {"names": names, "vectors": torch.from_numpy(vectors)}


def concept_vector(bank, name):
    if name not in bank["names"]:
        raise KeyError(f"concept '{name}' not in bank")
    idx = bank["names"].index(name)
    return bank["vectors"][idx]


def rank_concepts(pcbm, class_idx, k=10):
    weights = pcbm.classifier.weight.detach().cpu()
    values, indices = torch.topk(weights[class_idx], k=min(k, weights.shape[1]))
    return [(pcbm.names[i], float(v)) for i, v in zip(indices.tolist(), values.tolist())]


def gradcam(model, image, class_idx, target_layer=None):
    if target_layer is None:
        target_layer = model.layer4[-1]
    activations = {}
    gradients = {}

    def forward_hook(module, inputs, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)
    try:
        logits = model(image)
        model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()
    finally:
        handle_f.remove()
        handle_b.remove()
    acts = activations["value"][0].detach()
    grads = gradients["value"][0].detach()
    weights = grads.mean(dim=(1, 2), keepdim=True)
    cam = F.relu((weights * acts).sum(dim=0))
    cam = cam / (cam.max() + 1e-8)
    return cam.cpu().numpy()


def load_clip(device="cpu", model_name="RN50"):
    import clip
    model, _ = clip.load(model_name, device=device)
    return model.eval()


@torch.no_grad()
def clip_text_vector(clip_model, text):
    import clip
    device = next(clip_model.parameters()).device
    tokens = clip.tokenize([text]).to(device)
    vector = clip_model.encode_text(tokens).float()
    return vector / vector.norm(dim=-1, keepdim=True)


def clip_normalize(batch):
    mean = torch.tensor(CLIP_MEAN, device=batch.device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=batch.device).view(1, 3, 1, 1)
    return (batch - mean) / std


@torch.no_grad()
def patch_similarity(clip_model, image, vectors, patch=64, stride=16, batch=256):
    device = next(clip_model.parameters()).device
    img = image.to(device)
    patches = img.unfold(2, patch, stride).unfold(3, patch, stride)
    grid_h, grid_w = patches.shape[2], patches.shape[3]
    patches = patches.reshape(-1, 3, patch, patch)
    patches = F.interpolate(patches, size=(224, 224), mode="bilinear", align_corners=False)
    features = clip_model.encode_image(clip_normalize(patches)).float()
    features = features / features.norm(dim=-1, keepdim=True)
    vecs = vectors.to(device).float()
    chunks = []
    for start in range(0, features.shape[0], batch):
        chunks.append(features[start:start + batch] @ vecs.T)
    sims = torch.cat(chunks, dim=0)
    return sims.reshape(grid_h, grid_w, -1).cpu().numpy(), grid_h, grid_w


def locate_peak(similarity_map, stride, patch):
    score = similarity_map.reshape(similarity_map.shape[0], similarity_map.shape[1], -1)
    flat = score.argmax()
    grid_w = similarity_map.shape[1]
    vec_idx = int(flat % score.shape[2])
    spatial = int(flat // score.shape[2])
    row = spatial // grid_w
    col = spatial % grid_w
    center = (col * stride + patch // 2, row * stride + patch // 2)
    peak = float(score.max())
    return center, vec_idx, peak


def crop_patch(image, center, size):
    _, _, height, width = image.shape
    x, y = center
    half = size // 2
    left = min(max(x - half, 0), width - size)
    top = min(max(y - half, 0), height - size)
    left = max(left, 0)
    top = max(top, 0)
    return image[:, :, top:top + size, left:left + size]


def paste_patch(base, patch, center):
    _, _, height, width = base.shape
    size = patch.shape[-1]
    x, y = center
    half = size // 2
    left = min(max(x - half, 0), width - size)
    top = min(max(y - half, 0), height - size)
    left = max(left, 0)
    top = max(top, 0)
    out = base.clone()
    out[:, :, top:top + size, left:left + size] = patch
    return out
