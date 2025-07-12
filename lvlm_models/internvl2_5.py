import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

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

    def __call__(self, question, img_files):
        all_pixel_values = []
        for path in img_files:
            pixels = load_image(path, input_size=self.input_size)
            all_pixel_values.append(pixels)

        pixel_values = torch.cat(all_pixel_values, dim=0).to(self.dtype).to(self.device)
        response, _ = self.model.chat(self.tokenizer, 
                                      pixel_values, 
                                      question, 
                                      self.generation_config, 
                                      return_history=True
                                      )
        return response
    
    def compute_log_prob(self, question, img_files, answer):
        
        all_pixel_values = []
        for img in img_files:
            pixels = load_image(img, input_size=self.input_size)
            all_pixel_values.append(pixels)
        
        pixel_values = torch.cat(all_pixel_values, dim=0).to(self.dtype).to(self.device)
        
        answer_tokens = self.tokenizer(answer, return_tensors='pt', add_special_tokens=False).input_ids.to(self.device)
        answer_token_list = answer_tokens[0].tolist()
        
        with torch.no_grad():
            messages = [{'role': 'user', 'content': question}]
            
            temp_response, history = self.model.chat(
                self.tokenizer, 
                pixel_values, 
                question, 
                dict(max_new_tokens=1, do_sample=False),
                return_history=True
            )
            
            conversation_text = ""
            for turn in history:
                if turn['role'] == 'user':
                    conversation_text += f"User: {turn['content']}\n"
                elif turn['role'] == 'assistant':
                    conversation_text += f"Assistant: "
                    break
            
            input_ids = self.tokenizer(conversation_text, return_tensors='pt').input_ids.to(self.device)
            
            generation_config = dict(
                max_new_tokens=len(answer_token_list),
                min_new_tokens=len(answer_token_list),
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
            
            outputs = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                **generation_config
            )
            
            scores = outputs.scores  # List of tensors, mỗi tensor là logits cho 1 generation step
            total_log_prob = 0.0
            
            print(f"Number of scores: {len(scores)}")
            print(f"Number of target tokens: {len(answer_token_list)}")
            
            for i, score in enumerate(scores):
                if i < len(answer_token_list):
                    log_probs = F.log_softmax(score, dim=-1)
                    target_token = answer_token_list[i]
                    token_log_prob = log_probs[0, target_token].item()
                    total_log_prob += token_log_prob
                    
                    predicted_token = torch.argmax(score, dim=-1)[0].item()
                    print(f"Step {i}: target={target_token}, predicted={predicted_token}, log_prob={token_log_prob:.4f}")
            
            
        
        return total_log_prob
        
    
def add_gaussian_noise(img, std=0.1):
    transform_to_tensor = T.ToTensor()
    transform_to_pil = T.ToPILImage()

    tensor_img = transform_to_tensor(img)
    noise = torch.randn(tensor_img.size()) * std
    noisy_img = tensor_img + noise
    noisy_img = torch.clamp(noisy_img, 0, 1)

    return transform_to_pil(noisy_img)
    
if __name__ == "__main__":
    question = "Discribe these images. <IMAGE_TOKEN><IMAGE_TOKEN><IMAGE_TOKEN>"
    img_files = [Image.open(f"test_{i + 1}.jpg").convert("RGB") for i in range(3)]

    lvlm = InternVL("InternVL2_5-8B")
    answer = lvlm(question, img_files)
    print(answer)
    p_clean = lvlm.compute_log_prob(question, img_files, answer[0])

    std = 0.05  
    noisy_imgs = [add_gaussian_noise(img, std=std) for img in img_files]
    [noisy_img.save(f"test_{i + 1}_noisy.jpg") for i, noisy_img in enumerate(noisy_imgs)]
    adv_answer = lvlm(question, noisy_imgs)
    print(adv_answer)
    p_adv = lvlm.compute_log_prob(question, noisy_imgs, adv_answer[0])
    print(p_adv, p_clean)
    print(p_adv / p_clean)
    # print(math.exp(p_adv) / math.exp(p_clean))