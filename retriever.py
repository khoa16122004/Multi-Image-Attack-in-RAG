import torch
from PIL import Image
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
    