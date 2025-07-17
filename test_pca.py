from lvlm_models.llava_ import LLava
from util import DataLoader
import os
import torch
import os
import torch
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA
from torchvision.utils import save_image
import matplotlib.pyplot as plt

def save_patch_visualizations(patch_feats, retri_imgs, save_path="vis_patch_rgb/grid.jpg"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    B, N, D = patch_feats.shape
    side = int(N ** 0.5)

    fig, axs = plt.subplots(nrows=B, ncols=2, figsize=(6, 3 * B))

    if B == 1:
        axs = [axs]  # Ensure axs[i][j] is accessible

    for i in range(B):
        # Cột 1: Ảnh gốc
        img = retri_imgs[i]
        axs[i][0].imshow(img)
        axs[i][0].set_title("Original")
        axs[i][0].axis("off")

        # Cột 2: PCA patch -> ảnh
        pca = PCA(n_components=3)
        patch_rgb = pca.fit_transform(patch_feats[i].cpu().numpy())  # (N, 3)
        patch_rgb = patch_rgb.reshape(side, side, 3)

        patch_rgb = (patch_rgb - patch_rgb.min()) / (patch_rgb.max() - patch_rgb.min() + 1e-5)

        axs[i][1].imshow(patch_rgb)
        axs[i][1].set_title("Patch PCA")
        axs[i][1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# data
result_clean_dir = "result_usingquery=0_clip"
loader = DataLoader(retri_dir=result_clean_dir)
sample_path = "run.txt"
with open(sample_path, "r") as f:
    sample_ids = [int(line.strip()) for line in f]

# model
instruction = "Answer the given question based only on the visual content of the images. Do not guess or use outside knowledge. Just return the answer and nothing else."
model = LLava(
    pretrained="llava-next-interleave-qwen-7b",
    model_name="llava_qwen",
)

image_token = "<image>"

# sample
sample_index = 10
question, answer, query, gt_basenames, retri_basenames, retri_imgs, sims = loader.take_retri_data(sample_ids[sample_index])
patch_feats = model.extract_patch_features(retri_imgs)
print(patch_feats.shape)  # Should print the shape of the patch features tensor

save_patch_visualizations(patch_feats, retri_imgs)

