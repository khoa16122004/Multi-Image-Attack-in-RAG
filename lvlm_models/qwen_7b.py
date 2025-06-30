from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import math
import copy

class Qwen2VL:
    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.model.eval()

    def __call__(self, question, img_files, num_return_sequences=1, do_sample=False, temperature=0.7):
        # Prepare message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img} for img in img_files
                ] + [{"type": "text", "text": question}],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=do_sample,
                temperature=temperature,
                num_return_sequences=num_return_sequences
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        return output_text

    def compute_log_prob(self, question, imgs, answer):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img} for img in imgs
                ] + [{"type": "text", "text": question}],
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss

        total_log_prob = -loss.item() * labels.shape[1]
        prob = math.exp(total_log_prob)
        return prob
