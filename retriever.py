import torch
from PIL import Image
from util import DataLoader
import os
import json
import pickle

class Retriever(torch.nn.Module):
    def __init__(self, model_name='clip'):
        super().__init__()
        if model_name == "clip":
            from vl_models import CLIPModel
            self.model = CLIPModel()
        elif model_name == "blip":
            from vl_models import BLIPModel
            self.model = BLIPModel()
            
    @torch.no_grad()
    def forward(self, qs, img_files):
        adv_embeds = self.model.extract_visual_features(img_files)
        query_embedding = self.model.extract_textual_features([qs])
        sim = adv_embeds @ query_embedding.T
        
        return sim
    
if __name__ == "__main__":
    # sample_id
    sample_id = 1
    n_k = 3
    # path
    retri_dir = "retri_result_clip"
    reader_dir = "reader_result/Llama-7b/clip"
    result_dir = f"attack_result"
    retriever_name = "clip"
    reader_name = "llava"
    std = 0.1

    
    # model
    retriever = Retriever(retriever_name=retriever_name)
    
    # data
    loader = DataLoader(retri_dir=retri_dir)
    question, answer, query, gt_basenames, retri_basenames, retri_imgs = loader.take_data(sample_id)
    
    # init fitness data
    print(retriever(query, retri_imgs))