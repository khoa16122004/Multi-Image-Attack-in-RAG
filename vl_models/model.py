from PIL import Image
from typing import List

class VisionModel:
    def __init__(self, model_name: str, pretrained: str):
        self.model_name = model_name
        self.pretrained = pretrained

    def load_model(self):
        pass
    def extract_visual_features(self, imgs: List[Image.Image]):
        pass
    def extract_textual_features(self, texts: List[str]):
        pass