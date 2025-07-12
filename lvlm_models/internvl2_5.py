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
        response, _ = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config, return_history=True)
        return response
    
    def compute_log_prob(self, question, img_files, answer):
        """
        Tính log probability của answer given image và question
        
        Args:
            question: Câu hỏi dạng string
            img_files: List các PIL Image objects
            answer: Câu trả lời cần tính probability
        
        Returns:
            log_prob: Log probability của answer
        """
        import torch.nn.functional as F
        
        # Preprocess images giống như trong __call__
        all_pixel_values = []
        for img in img_files:
            pixels = load_image(img, input_size=self.input_size)
            all_pixel_values.append(pixels)
        
        pixel_values = torch.cat(all_pixel_values, dim=0).to(self.dtype).to(self.device)
        
        # Tokenize question và answer
        # Tạo prompt giống như model.chat sử dụng
        messages = [{'role': 'user', 'content': question}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize input và target
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)
        target_ids = self.tokenizer(answer, return_tensors='pt').input_ids.to(self.device)
        
        # Tạo full sequence: input + target
        full_input_ids = torch.cat([input_ids, target_ids], dim=1)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=full_input_ids,
                pixel_values=pixel_values,
                return_dict=True
            )
            
            logits = outputs.logits
            
            # Tính log probability cho phần answer
            # Shift logits và labels để align
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = full_input_ids[..., 1:].contiguous()
            
            # Chỉ tính loss cho phần answer (bỏ qua phần prompt)
            answer_start_idx = input_ids.size(1) - 1  # -1 vì đã shift
            answer_logits = shift_logits[:, answer_start_idx:, :]
            answer_labels = shift_labels[:, answer_start_idx:]
            
            # Tính log probabilities
            log_probs = F.log_softmax(answer_logits, dim=-1)
            
            # Lấy log prob của các token trong answer
            token_log_probs = log_probs.gather(2, answer_labels.unsqueeze(-1)).squeeze(-1)
            
            # Tổng log probability của toàn bộ answer
            total_log_prob = token_log_probs.sum().item()
            
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
    question = "Discribe these images. <image><image><image>"
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