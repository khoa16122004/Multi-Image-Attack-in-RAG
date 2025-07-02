from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
import torch
import math
from PIL import Image

class DeepSeekVL2:
    def __init__(self, pretrained):
        model_path = f"deepseek-ai/{pretrained}"
        self.vl_chat_proccessor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_proccessor.tokenizer

        self.vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()

    def __call__(self, qs, img_files):
        conversation = [
            {
                "role": "User",
                "content": qs,
                "images": img_files
            },
            {"role": "Assistant", "content": ""}
        ]

        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(vl_gpt.device)      

        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        with torch.inference_mdoe():

            cont = vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True
            )

        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        outputs = text_outputs
        return outputs

if __name__ == "__main__":
    question = "What is the shape of nostrils on bill of the Russet-naped Wood-Rail (scientific name: Aramides albiventris)? <image_placeholder><image_placeholder><image_placeholder>"
    img_files = [Image.open(f"test_{i + 1}.jpg") for i in range(3)]

    
    lvlm = DeepSeekVL2("deepseek-vl2-small")
    answer = lvlm(qs, img_files)
    print(answer)