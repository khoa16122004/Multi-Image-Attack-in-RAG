import os
import sys
import torch
from PIL import Image
import pickle as pkl
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from util import DataLoader
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
        elif model_name == "llava-one":
            from lvlm_models.llava_ import LLava
            self.instruction = "Answer the given question based only on the visual content of the images. Do not guess or use outside knowledge. Keep the answer short."
            self.model = LLava(
                pretrained="llava-onevision-qwen2-7b-ov",
                model_name="llava_qwen",
            )
            
    def init_data(self, golden_answer):
        self.answer = golden_answer
        # self.gt_embedding = self.clip_model.extract_textual_features([golden_answer])[0]
    
    def compute_similarity(self, preds):
        pred_embeddings = self.clip_model.extract_textual_features(preds)
        sim = pred_embeddings @ self.gt_embedding.T
        return sim
        
          
    @torch.no_grad()
    def image_to_text(self, qs, img_files):
        prompt = f"{self.instruction}\n question: {qs}\n images: " + "<image>" * len(img_files)
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
    reader = Reader()
    question, answer, query, gt_basenames, retri_basenames, retri_imgs = DataLoader(retri_dir="retri_result_clip").take_data(0)
    adv_img_tensor = transforms.ToTensor()(retri_imgs[1]).cuda() + torch.rand(3, 312, 312).cuda() * 100
    reader.init_data("The larval body of the carpenterworm moth is black and white.")
    scores = reader(question, [retri_imgs])
    print(scores)
    
    retri_imgs[1] = to_pil_image(adv_img_tensor)
    scores = reader(question, [retri_imgs])
    print(scores)
    

            
            

        