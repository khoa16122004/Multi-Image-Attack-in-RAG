from lvlm_models.llava_ import LLava
from util import DataLoader
import os
import torch

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
