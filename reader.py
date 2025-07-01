import os
import sys
import torch
from PIL import Image
import pickle as pkl
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from vl_models import CLIPModel

class Reader(torch.nn.Module):
    def __init__(self, model_name="llava"):
        super().__init__()
        self.clip_model =  CLIPModel()
        
        if model_name == "llava-next":
            from lvlm_models.llava_ import LLava
            self.instruction = "Answer the given question based only on the visual content of the images. Do not guess or use outside knowledge. Keep the answer short."
            self.model = LLava(
                pretrained="llava-next-interleave-qwen-7b",
                model_name="llava_qwen",
            )
            self.image_token = "<image>"
            
        elif model_name == "llava-one":
            from lvlm_models.llava_ import LLava
            self.instruction = "Answer the given question based only on the visual content of the images. Do not guess or use outside knowledge."
            self.model = LLava(
                pretrained="llava-onevision-qwen2-7b-ov",
                model_name="llava_qwen",
            )
            self.image_token = "<image>"

        elif model_name == "deepseek-vl":
            from lvlm_models.deepseek_ import DeepSeek
            self.instruction = "Answer the given question based only on the visual content of the images. Do not guess or use outside knowledge."
            self.model = DeepSeek(
                pretrained="DeepSeek-VL-7B-chat",
            )
            self.image_token = "<image_placeholder>"

            
        elif model_name == "mantis":
            from lvlm_models.mantis_ import Mantis
            self.instruction = "Answer the given question based only on the visual content of the images. Do not guess or use outside knowledge."
            self.model = Mantis(
                pretrained="Mantis-llava-7b",
            )
            self.image_token = "<image>"

            
    def init_data(self, golden_answer):
        self.answer = golden_answer
        # self.gt_embedding = self.clip_model.extract_textual_features([golden_answer])[0]
    
    def compute_similarity(self, preds):
        pred_embeddings = self.clip_model.extract_textual_features(preds)
        sim = pred_embeddings @ self.gt_embedding.T
        return sim
        
          
    @torch.no_grad()
    def image_to_text(self, qs, img_files):
        prompt = f"{self.instruction}\n question: {qs}\n images: " + self.image_token * len(img_files)
        text_output = self.model(prompt, img_files)  # string output
        return text_output
    
    @torch.no_grad()
    def forward(self, qs, img_files):
        prompt = f"{self.instruction}\n question: {qs}\n images:" + len(img_files[0]) * "<image>"
        all_outputs = []

        for topk_imgs in img_files:
            score = self.model.compute_log_prob(prompt, topk_imgs, self.answer)
            all_outputs.append(score)

        scores = torch.tensor(all_outputs)
        return scores.cuda()


if __name__ == "__main__":
    Reader("mantis")
    

    

            
            

        