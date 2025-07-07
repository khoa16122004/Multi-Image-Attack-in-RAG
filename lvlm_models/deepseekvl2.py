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
        vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()


    def __call__(self, qs, img_files):
        conversation = [
            {
                "role": "<|User|>",
                "content": qs,
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

        prepare_inputs = self.vl_chat_proccessor(
            conversations=conversation,
            images=img_files,
            force_batchify=True,
            system_prompt=""
        ).to(self.vl_gpt.device)      

        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)


        cont = self.vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        outputs = self.tokenizer.decode(cont[0].cpu().tolist(), skip_special_tokens=False).split("<｜end▁of▁sentence｜>")
        return outputs
    
    def compute_log_prob(self, question, img_files, answer):
        conversation = [
            {"role": "<|User|>", "content": question},
            {"role": "<|Assistant|>", "content": answer},
        ]
        pil_images = [Image.open(f) if isinstance(f, str) else f for f in img_files]
        prepare = self.vl_chat_proccessor(conversations=conversation, images=pil_images, force_batchify=True, system_prompt="").to(self.vl_gpt.device)

        prompt_conv = [
            {"role": "<|User|>", "content": question},
            {"role": "<|Assistant|>", "content": ""}
        ]
        prompt_ids = self.tokenizer.apply_chat_template(prompt_conv, tokenize=True, add_generation_prompt=False, return_tensors="pt").to(self.vl_gpt.device)
        q_len = prompt_ids.shape[1]

        labels = prepare.input_ids.clone()
        labels[:, :q_len] = -100

        with torch.no_grad():
            out = self.vl_gpt(input_ids=prepare.input_ids, 
                              attention_mask=prepare.attention_mask, 
                              labels=labels,
                              use_cache=True
                              )
            loss = out.loss.item()
        num_tokens = (labels != -100).sum().item()
        total_log_prob = -loss * num_tokens
        return math.exp(total_log_prob)

    
if __name__ == "__main__":
    question = "What is the shape of nostrils on bill of the Russet-naped Wood-Rail (scientific name: Aramides albiventris)? <image_placeholder> <image_placeholder> <image_placeholder>"
    img_files = [Image.open(f"test_{i + 1}.jpg") for i in range(3)]

    
    lvlm = DeepSeekVL2("deepseek-vl2-tiny")
    prob_1 = lvlm.compute_log_prob(question, img_files, "The shape of the nostrils on the bill of the Russet-naped Wood-Rail (Aramides albiventris) is not clearly visible in the provided images. However, based on the scientific name and the general appearance of the bird, it is likely that the nostrils are located on the upper mandible, which is the upper part of the beak. The upper mandible of the Russet-naped Wood-Rail is typically long and slender, with a slightly curved shape. The nostrils are usually located near the tip of the upper mandible, and they are typically small and oval-shaped.")
    print(prob_1)
    prob_2 = lvlm.compute_log_prob(question, img_files, "a bird")
    print(prob_2)
