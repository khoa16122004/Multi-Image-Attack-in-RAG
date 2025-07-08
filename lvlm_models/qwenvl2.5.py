import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from typing import List, Union
from PIL import Image
import math
import torchvision.transforms as T

class QwenVL:
    def __init__(self, model_name="Qwen2.5-VL-7B-Instruct", device="cuda"):
        self.device = device
        self.model_path = f"Qwen/{model_name}"
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_path, trust_remote_code=True).to(device)
        self.model.eval()
    
    def __call__(self, qs: str, img_files, num_return_sequences=1, do_sample=False, temperature=0.0):
        messages = [
            {
                "role": "user",
                "content": [],
            }
        ]
        # Add contents
        for img_file in img_files:
            messages[0]["content"].append({"type": "image", "image": img_file})
        messages[0]["content"].append({"type": "text", "text": qs})

        # Prepare inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=do_sample,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

    def compute_log_prob(self, question: str, imgs, answer: str):
        messages = [{
            "role": "user",
            "content": [],
        }]
        for img_file in imgs:
            if isinstance(img_file, str):
                img = Image.open(img_file).convert("RGB")
            else:
                img = img_file
            messages[0]["content"].append({"type": "image", "image": img})
        messages[0]["content"].append({"type": "text", "text": question})

        # Prepare input
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text + answer],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        labels = inputs.input_ids.clone()
        labels[:, :-len(self.processor.tokenizer(answer)["input_ids"])] = -100  # Only answer is supervised

        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            num_tokens = (labels != -100).sum().item()
            log_prob = -loss.item() * num_tokens
            prob = math.exp(log_prob)
        return prob

    def _process_vision_info(self, messages):
        images = []
        videos = []
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image":
                    images.append(content["image"])
                elif content["type"] == "video":
                    videos.append(content["video"])
        return images, videos
    
    
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

    lvlm = QwenVL("Qwen2.5-VL-7B-Instruct")
    answer = lvlm(question, img_files)
    # print(lvlm.compute_log_prob(question, img_files, answer[0]))
    print(answer)

    # Add noise
    std = 0.05  # Bạn có thể thử các giá trị như 0.05, 0.1, 0.2
    noisy_imgs = [add_gaussian_noise(img, std=std) for img in img_files]
    [noisy_img.save(f"test_{i + 1}_noisy.jpg") for i, noisy_img in enumerate(noisy_imgs)]
    adv_answer = lvlm(question, noisy_imgs)
    print(adv_answer)
    # print(lvlm.compute_log_prob(question, noisy_imgs, answer[0]))
