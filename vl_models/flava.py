from transformers import FlavaModel, FlavaProcessor
from PIL import Image
from typing import List
from .model import VisionModel
import torch

class FLAVAModel(VisionModel):
    def __init__(self, model_name="facebook/flava-full", pretrained="facebook"):
        super().__init__(model_name, pretrained)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        self.processor = FlavaProcessor.from_pretrained(self.model_name)
        self.model = FlavaModel.from_pretrained(self.model_name).to(self.device)

    def extract_visual_features(self, imgs: List[Image.Image]):
        inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        return torch.nn.functional.normalize(outputs, p=2, dim=-1)

    def extract_textual_features(self, texts: List[str]):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        return torch.nn.functional.normalize(outputs, p=2, dim=-1)
