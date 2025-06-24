from transformers import BlipProcessor, BlipModel
import torch
from PIL import Image
from typing import List
from .model import VisionModel

class BLIPModel(VisionModel):
    def __init__(self, model_name="Salesforce/blip-image-captioning-base", pretrained="salesforce"):
        super().__init__(model_name, pretrained)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
        self.model_name = "blip"

    def load_model(self):
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipModel.from_pretrained(self.model_name).to(self.device)

    def extract_visual_features(self, imgs: List[Image.Image]):
        inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.vision_model(**inputs)
        features = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return torch.nn.functional.normalize(features, p=2, dim=-1)

    def extract_textual_features(self, texts: List[str]):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.text_model(**inputs)
        features = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return torch.nn.functional.normalize(features, p=2, dim=-1)


# BLIPModel()