import torch
import numpy as np
import math
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from lmdeploy.vl.constants import IMAGE_TOKEN


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(set((i, j) for n in range(min_num, max_num + 1)
                               for i in range(1, n + 1) for j in range(1, n + 1)
                               if min_num <= i * j <= max_num), key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = [
        resized_img.crop((
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        ))
        for i in range(blocks)
    ]
    if use_thumbnail and blocks != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(image) for image in images])
    return pixel_values

def add_gaussian_noise(img, std=0.1):
    transform_to_tensor = T.ToTensor()
    transform_to_pil = T.ToPILImage()

    tensor_img = transform_to_tensor(img)
    noise = torch.randn(tensor_img.size()) * std
    noisy_img = tensor_img + noise
    noisy_img = torch.clamp(noisy_img, 0, 1)

    return transform_to_pil(noisy_img)

class InternVL:
    def __init__(self, model_name="InternVL2_5-8B", input_size=448):
        model_path = f"OpenGVLab/{model_name}"
        self.device = 'cuda'
        self.dtype = torch.bfloat16
        self.input_size = input_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True
        ).eval().to(self.device)

        self.generation_config = dict(max_new_tokens=1024, do_sample=True)

    def _prepare_pixel_values(self, img_files):
        """Process images and return pixel values"""
        all_pixel_values = []
        for img in img_files:
            pixels = load_image(img, input_size=self.input_size)
            all_pixel_values.append(pixels)
        return torch.cat(all_pixel_values, dim=0).to(self.dtype).to(self.device)

    def __call__(self, question, img_files):
        pixel_values = self._prepare_pixel_values(img_files)
        response, _ = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config, return_history=True)
        return response
    
    def compute_log_prob(self, question, img_files, answer):
        pixel_values = self._prepare_pixel_values(img_files)
        prompt_input_ids = self.tokenizer.encode(question, return_tensors='pt').to(self.device)
        full_text = question + answer
        full_input_ids = self.tokenizer.encode(full_text, return_tensors='pt').to(self.device)
        
        attention_mask = torch.ones_like(full_input_ids)
        
        labels = full_input_ids.clone()
        prompt_len = prompt_input_ids.shape[1]
        labels[0, :prompt_len] = -100
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=full_input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                return_dict=True
            )
            
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            
            valid_mask = (shift_labels != -100)
            total_log_prob = token_log_probs[valid_mask].sum().item()
            return total_log_prob


if __name__ == "__main__":
    question = f"Describe these images. {IMAGE_TOKEN} {IMAGE_TOKEN} {IMAGE_TOKEN}"
    img_files = [Image.open(f"test_{i + 1}.jpg").convert("RGB") for i in range(3)]

    lvlm = InternVL("InternVL2_5-8B")
    
    answer = lvlm(question, img_files)
    print("Clean answer:", answer)
    
    p_clean = lvlm.compute_log_prob(question, img_files, answer[0])
    print("Clean log prob:", p_clean)

    std = 0.05  
    noisy_imgs = [add_gaussian_noise(img, std=std) for img in img_files]
    [noisy_img.save(f"test_{i + 1}_noisy.jpg") for i, noisy_img in enumerate(noisy_imgs)]
    
    adv_answer = lvlm(question, noisy_imgs)
    print("Noisy answer:", adv_answer)
    
    p_adv = lvlm.compute_log_prob(question, noisy_imgs, adv_answer[0])
    print("Noisy log prob:", p_adv)
    
    print(f"Log prob ratio: {p_adv / p_clean if p_clean != 0 else 'inf'}")
    