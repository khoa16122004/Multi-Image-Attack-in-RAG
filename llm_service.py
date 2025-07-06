import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, MistralForCausalLM
import os
from openai import OpenAI
from typing import List, Optional
import dotenv
dotenv.load_dotenv()
class QwenService:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def text_to_text(self, system_prompt, prompt):
        inputs = f"{system_prompt}\n{prompt}"
        input_ids = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
        outputs = self.model.generate(input_ids, max_new_tokens=50)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class GPTService:
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize the GPTService with a model name and API key.
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set in the environment variable 'OPENAI_KEY'.")
        self.client = OpenAI(api_key=self.api_key)

    def text_to_text(self, prompt: str, system_prompt: str) -> str:
        """
        Perform a text-to-text API call.
        """
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.output_text.strip()
        except Exception as e:
            print(f"Error during API call: {e}")
            return "Error occurred during API call."

    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:  
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def image_to_text(self, prompt: str, image_paths: List[str], system_prompt: str) -> str:
        """
        Perform an image-to-text API call using base64-encoded images.
        """
        try:
            base64_images = [self.encode_image(image_path) for image_path in image_paths]
            input_images = [
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}
                for b64 in base64_images
            ]

            response = self.client.responses.create(
                model=self.model_name,
                input=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt}] + input_images
                    }
                ]
            )
            return response.output_text.strip()
        except Exception as e:
            print(f"Error during API call: {e}")
            return "Error occurred during API call."

from transformers import AutoModelForCausalLM, AutoTokenizer

class LlamaService:
    def __init__(self, model_name):
        self.cls_mapping = {
            "Llama-7b": (LlamaForCausalLM, LlamaTokenizer, True, "Llama-2-7b-chat-hf", "meta-llama"),
            "Llama-13b": (LlamaForCausalLM, LlamaTokenizer, True, "Llama-2-13b-chat-hf", "meta-llama"),
            "Mistral-7b": (MistralForCausalLM, AutoTokenizer, True, "Mistral-7B-Instruct-v0.2", ""),
            "vicuna-7b": (LlamaForCausalLM, LlamaTokenizer, True, "vicuna-7b-v1.5", ""),
            "vicuna-13b": (LlamaForCausalLM, LlamaTokenizer, True, "vicuna-13b-v1.5", ""),
            "gemma-7b": (AutoModelForCausalLM, AutoTokenizer, True, "gemma-7b-it", ""),
            "llama-3-8b": (AutoModelForCausalLM, AutoTokenizer, True, "Meta-Llama-3-8B-Instruct", "meta-llama")  # new
        }

        self.model_name = model_name
        model_cls, tokenizer_cls, self.is_decoder, hf_name, prefix = self.cls_mapping[model_name]

        model_path = os.path.join(prefix, hf_name) if prefix else hf_name

        self.model = model_cls.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
        self.tokenizer = tokenizer_cls.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.generate_kwargs = dict(
            max_new_tokens=512,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_scores=True,
            do_sample=False,
            top_p=None,
            temperature=None,
        )

        if self.is_decoder:
            self.tokenizer.padding_side = "left"

    def text_to_text(self, system_prompt, prompt):
        inputs = f"[INST]{system_prompt}\n{prompt}[/INST]Answer: "
        input_ids = self.tokenizer(inputs,
                                   return_tensors="pt",
                                   padding=True,
                                   truncation=True)
        outputs = self.model.generate(
            input_ids=input_ids.input_ids.to(self.model.device),
            attention_mask=input_ids.attention_mask.to(self.model.device),
            **self.generate_kwargs
        )
        decoded_outputs = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        if isinstance(decoded_outputs, list):
            return [o.split("Answer:")[-1].strip() for o in decoded_outputs]
        else:
            return decoded_outputs.split("Answer:")[-1].strip()
