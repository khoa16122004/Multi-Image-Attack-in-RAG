from mantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration
from mantis.models.mllava import chat_mllava
from mantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration
processor = MLlavaProcessor.from_pretrained("TIGER-Lab/Mantis-8B-clip-llama3")

from huggingface_hub import hf_hub_download
import torch

class Mantis:
    def __init__(self, pretrained):
        # Mantis-8B-clip-llama3
        # Mantis-8B-siglip-llama3
        self.processor = MLlavaProcessor.from_pretrained(f"TIGER-Lab/{pretrained}")
        self.model = LlavaForConditionalGeneration.from_pretrained(f"TIGER-Lab/{pretrained}", device_map=f"cuda:{torch.cuda.current_device()}", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

        self.generation_kwargs = {"max_new_tokens": 1024, "num_beams": 1, "do_sample": False}
    def inference(self, qs, img_files, num_return_sequences=1, do_sample=True, temperature=0, reload=True):
        if not do_sample and num_return_sequences > 1:
            raise ValueError("Greedy decoding doesn't support multiple return sequences. Set do_sample=True or num_beams > 1.")

        response, history = chat_mllava(qs, img_files, 
                                        self.model, 
                                        self.processor, 
                                        do_sample=do_sample,
                                        temperature=temperature,
                                        max_new_tokens=4096,
                                        num_return_sequences=num_return_sequences)
        
        return [response]

        
        
        