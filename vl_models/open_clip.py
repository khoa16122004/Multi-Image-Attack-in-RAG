import open_clip
import torch
from PIL import Image
from typing import List
from .model import VisionModel

class OpenCLIPModel(VisionModel):
    def __init__(self, model_name="ViT-H-14", pretrained="laion2b_s32b_b79k"):
        super().__init__(model_name, pretrained)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self.model = self.model.to(self.device)

    def extract_visual_features(self, imgs: List[Image.Image]):
        imgs = torch.stack([self.preprocess(img) for img in imgs]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(imgs)
        return torch.nn.functional.normalize(features, p=2, dim=-1)

    def extract_textual_features(self, texts: List[str]):
        tokens = open_clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
        return torch.nn.functional.normalize(features, p=2, dim=-1)