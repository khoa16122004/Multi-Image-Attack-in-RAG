import torch
import os
from PIL import Image
from typing import List
from .model import VisionModel
import clip
from dotenv import load_dotenv

load_dotenv()
os.environ['CURL_CA_BUNDLE'] = ''


class CLIPModel(VisionModel):
    def __init__(self, model_name: str = "clip", pretrained: str = "ViT-B/32"):
        super().__init__(model_name, pretrained)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        self.model, self.preprocess = clip.load(self.pretrained, device=self.device, jit=False)
        self.model.eval()
    def extract_visual_features(self, imgs: List[Image.Image]):
        images = torch.stack([self.preprocess(img) for img in imgs]).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        return image_features

    def extract_textual_features(self, texts: List[str]):
        tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
        return text_features
