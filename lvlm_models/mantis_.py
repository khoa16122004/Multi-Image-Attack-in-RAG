import torch
import math
from mantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration
from mantis.models.mllava import chat_mllava

class Mantis:
    def __init__(self, pretrained):
        self.processor = MLlavaProcessor.from_pretrained(f"TIGER-Lab/{pretrained}")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            f"TIGER-Lab/{pretrained}",
            device_map=f"cuda:{torch.cuda.current_device()}",
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2"
        )
        self.model.eval()
        self.generation_kwargs = {
            "max_new_tokens": 1024,
            "num_beams": 1,
            "do_sample": False
        }

        # Gán thủ công chat_template để chat_mllava không bị lỗi
        self.processor.tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ bos_token }}User: {{ message['content'][0]['text'] }}\n"
            "{% elif message['role'] == 'assistant' %}"
            "Assistant: {{ message['content'][0]['text'] }}\n"
            "{% endif %}"
            "{% endfor %}"
        )

    def __call__(self, qs, img_files, num_return_sequences=1, do_sample=False, temperature=0):
        if not do_sample and num_return_sequences > 1:
            raise ValueError("Greedy decoding doesn't support multiple return sequences. Set do_sample=True or num_beams > 1.")

        response, history = chat_mllava(
            qs, img_files,
            self.model,
            self.processor,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=4096,
            num_return_sequences=num_return_sequences
        )
        return [response]

    def compute_log_prob(self, question, imgs, answer):
        # Tự tạo prompt thủ công thay vì dùng apply_chat_template
        full_prompt = f"User: {question}\nAssistant: {answer}"
        encoding = self.processor(
            text=full_prompt,
            images=imgs,
            return_tensors="pt",
            padding="longest"
        ).to(self.model.device)

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        pixel_values = encoding["pixel_values"]

        # Tạo label chỉ cho phần trả lời
        labels = input_ids.clone()
        answer_ids = self.processor.tokenizer(answer, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(self.model.device)
        labels[:, :-answer_ids.shape[0]] = -100  # chỉ tính loss trên phần answer

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels
            )
            loss = outputs.loss

        num_answer_tokens = answer_ids.shape[0]
        total_log_prob = -loss.item() * num_answer_tokens
        prob = math.exp(total_log_prob)

        return prob
