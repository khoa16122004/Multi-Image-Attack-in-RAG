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
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        self.model.eval()
        self.generation_kwargs = {
            "max_new_tokens": 1024,
            "num_beams": 1,
            "do_sample": False
        }

    def __call__(self, qs, img_files, num_return_sequences=1, do_sample=True, temperature=0):
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
        # Build the full prompt as in chat_mllava
        messages = [
            {"role": "user", "content": [{"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]

        # Tokenize with images
        input_data = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        encoding = self.processor(
            text=input_data,
            images=imgs,
            return_tensors="pt",
            padding="longest"
        ).to(self.model.device)

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        pixel_values = encoding["pixel_values"]

        # Create labels: ignore prompt tokens, only compute loss on answer
        labels = input_ids.clone()
        answer_ids = self.processor.tokenizer(answer, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(self.model.device)
        labels[:, :-answer_ids.shape[0]] = -100  # ignore question part

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
