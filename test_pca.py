import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from lvlm_models.llava_ import LLava
from util import DataLoader

def save_patch_visualizations(patch_feats, retri_imgs, save_path="vis_patch_rgb/grid.jpg", topk_percent=50):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    B, N, D = patch_feats.shape
    side = int(N ** 0.5)
    fig, axs = plt.subplots(nrows=B, ncols=2, figsize=(6, 3 * B))

    if B == 1:
        axs = [axs]  # Ensure 2D indexing

    for i in range(B):
        img = retri_imgs[i]
        axs[i][0].imshow(img)
        axs[i][0].set_title("Original")
        axs[i][0].axis("off")

        patch = patch_feats[i]  # (N, D)

        # ---- Lọc background ----
        norm = patch.norm(dim=1)  # (N,)
        k = int(N * topk_percent / 100)
        topk_indices = torch.topk(norm, k).indices
        mask = torch.zeros(N, dtype=torch.bool, device=patch.device)
        mask[topk_indices] = True
        patch_filtered = patch.clone()
        patch_filtered[~mask] = 0  # zero-out background

        # ---- PCA ----
        pca = PCA(n_components=3)
        patch_rgb = pca.fit_transform(patch_filtered.cpu().numpy())  # (N, 3)
        patch_rgb = patch_rgb.reshape(side, side, 3)
        patch_rgb = (patch_rgb - patch_rgb.min()) / (patch_rgb.max() - patch_rgb.min() + 1e-5)

        axs[i][1].imshow(patch_rgb)
        axs[i][1].set_title(f"Patch PCA ({topk_percent}%)")
        axs[i][1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# --- Load data ---
result_clean_dir = "result_usingquery=0_clip"
loader = DataLoader(retri_dir=result_clean_dir)
sample_path = "run.txt"

with open(sample_path, "r") as f:
    sample_ids = [int(line.strip()) for line in f]

# --- Load model ---
model = LLava(
    pretrained="llava-next-interleave-qwen-7b",
    model_name="llava_qwen",
)

# --- Visualize for each sample ---
for i in sample_ids:
    question, answer, query, gt_basenames, retri_basenames, retri_imgs, sims = loader.take_retri_data(i)
    patch_feats = model.extract_patch_features(retri_imgs)
    save_path = f"vis_patch_rgb/sample_{i}.jpg"
    save_patch_visualizations(patch_feats, retri_imgs, save_path=save_path, topk_percent=50)
