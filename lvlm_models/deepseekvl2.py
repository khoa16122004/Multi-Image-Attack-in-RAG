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

        outputs = self.tokenizer.decode(cont[0].cpu().tolist(), skip_special_tokens=False).split("<｜end▁of▁sentence｜>")[0]
        return outputs
    def compute_log_prob(self, question, img_files, answer):
        conversation = [
            {
                "role": "<|User|>",
                "content": question,
            },
            {
                "role": "<|Assistant|>",
                "content": answer,
            }
        ]

        prepare_inputs = self.vl_chat_proccessor(
            conversations=conversation,
            images=img_files,
            force_batchify=True,
            system_prompt=""
        ).to(self.vl_gpt.device)

        with torch.no_grad():
            outputs = self.vl_gpt(
                input_ids=prepare_inputs.input_ids,
                attention_mask=prepare_inputs.attention_mask,
                labels=prepare_inputs.input_ids,  # compute full loss,
                use_cache=True
            )
            loss = outputs.loss

        answer_only = [{"role": "<|Assistant|>", "content": answer}]
        answer_ids = self.vl_chat_proccessor.tokenizer.apply_chat_template(
            answer_only,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        ).to(self.vl_gpt.device)

        num_answer_tokens = answer_ids.shape[1]
        total_log_prob = -loss.item() * num_answer_tokens
        prob = math.exp(total_log_prob)
        return prob
    
if __name__ == "__main__":
    question = "What is the shape of nostrils on bill of the Russet-naped Wood-Rail (scientific name: Aramides albiventris)? <image_placeholder> <image_placeholder> <image_placeholder>"
    img_files = [Image.open(f"test_{i + 1}.jpg") for i in range(3)]

    
    lvlm = DeepSeekVL2("deepseek-vl2-small")
    prob = lvlm.compute_log_prob(question, img_files, "The shape of the nostrils on the bill of the Russet-naped Wood-Rail (Aramides albiventris) is not clearly visible in the provided images. However, based on the scientific name and the general appearance of the bird, it is likely that the nostrils are located on the upper mandible, which is the upper part of the beak. The upper mandible of the Russet-naped Wood-Rail is typically long and slender, with a slightly curved shape. The nostrils are usually located near the tip of the upper mandible, and they are typically small and oval-shaped.")
    print(prob)
    prob = lvlm.compute_log_prob(question, img_files, "a bird")
    print(prob)
