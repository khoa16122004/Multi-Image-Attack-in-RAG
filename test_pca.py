from lvlm_models.llava_ import LLava
from util import DataLoader
import os
import torch
import os
import torch
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA
from torchvision.utils import save_image

def save_patch_visualizations(patch_feats, retri_imgs, save_dir="vis_patch_rgb"):
    os.makedirs(save_dir, exist_ok=True)

    B, N, D = patch_feats.shape
    side = int(N ** 0.5)  # 27 nếu 729 patch

    for i in range(B):
        # Lấy ảnh PIL và convert sang tensor
        pil_img = retri_imgs[i]
        orig_img = TF.to_tensor(pil_img)  # (3, H, W)
        H_img, W_img = orig_img.shape[1:]

        # Giảm chiều bằng PCA
        pca = PCA(n_components=3)
        patch_rgb = pca.fit_transform(patch_feats[i].cpu().numpy())  # (729, 3)
        patch_rgb = torch.tensor(patch_rgb).T.float()  # (3, 729)

        # Reshape thành ảnh 3x27x27
        patch_rgb = patch_rgb.reshape(3, side, side)

        # Normalize về [0,1]
        patch_rgb = (patch_rgb - patch_rgb.min()) / (patch_rgb.max() - patch_rgb.min() + 1e-5)

        # Upsample thành ảnh cùng size
        patch_rgb = patch_rgb.unsqueeze(0)  # (1, 3, 27, 27)
        patch_rgb = torch.nn.functional.interpolate(patch_rgb, size=(H_img, W_img), mode='bilinear', align_corners=False).squeeze(0)

        # Nối ảnh gốc và ảnh PCA
        output = torch.cat([orig_img, patch_rgb], dim=-1)  # Nối theo chiều ngang

        # Lưu ảnh
        save_path = os.path.join(save_dir, f"{i}.jpg")
        save_image(output, save_path)


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

save_patch_visualizations(patch_feats, retri_imgs, save_dir="vis_patch_rgb")

