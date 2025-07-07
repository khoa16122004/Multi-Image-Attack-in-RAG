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
            {"role": "<|User|>", "content": question},
            {"role": "<|Assistant|>", "content": answer},
        ]

        prepare = self.vl_chat_proccessor(
            conversations=conversation,
            images=img_files,
            force_batchify=True,
            system_prompt=""
        ).to(self.vl_gpt.device)

        input_ids = prepare.input_ids
        attention_mask = prepare.attention_mask

        with torch.no_grad():
            outputs = self.vl_gpt(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=True,
            )

        logits = outputs.logits[:, :-1] 
        labels = input_ids[:, 1:]       

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs_answer = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

        answer_start = (input_ids[0] == self.tokenizer.convert_tokens_to_ids("<|Assistant|>")).nonzero(as_tuple=True)[0][0] + 1
        log_probs_answer_part = log_probs_answer[0, answer_start:]

        total_log_prob = log_probs_answer_part.sum().item()

        return math.exp(total_log_prob)  # = p(answer | question, imgs)

    
if __name__ == "__main__":
    question = "What is the shape of nostrils on bill of the Russet-naped Wood-Rail (scientific name: Aramides albiventris)? <image_placeholder> <image_placeholder> <image_placeholder>"
    img_files = [Image.open(f"test_{i + 1}.jpg") for i in range(3)]

    
    lvlm = DeepSeekVL2("deepseek-vl2-tiny")
    answer = lvlm(question, img_files)
    prob_1 = lvlm.compute_log_prob(question, img_files, answer)
    print(prob_1)
    prob_2 = lvlm.compute_log_prob(question, img_files, "a bird a bird a bird a bird a bird")
    print(prob_2)
