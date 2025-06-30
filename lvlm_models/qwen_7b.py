import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from typing import List, Union
from PIL import Image
import math

class QwenVL:
    def __init__(self, model_name="Qwen/Qwen-VL-Chat", device="cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.model.eval()
    
    def __call__(self, qs: str, img_files: List[Union[str, Image.Image]], num_return_sequences=1, do_sample=False, temperature=0.0):
        messages = [
            {
                "role": "user",
                "content": [],
            }
        ]
        # Add images
        for img_file in img_files:
            if isinstance(img_file, str):
                img = Image.open(img_file).convert("RGB")
            else:
                img = img_file
            messages[0]["content"].append({"type": "image", "image": img})

        # Add text
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

    def compute_log_prob(self, question: str, imgs: List[Union[str, Image.Image]], answer: str):
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
