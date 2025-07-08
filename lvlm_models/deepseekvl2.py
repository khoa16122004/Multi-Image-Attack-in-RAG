from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
import torch
import math
from PIL import Image
import torchvision.transforms as T

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
            use_cache=False
        )

        outputs = self.tokenizer.decode(cont[0].cpu().tolist(), skip_special_tokens=False)
        return [outputs]
    
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
                use_cache=False,
            )

        logits = outputs.logits[:, :-1] 
        labels = input_ids[:, 1:]       

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs_answer = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

        answer_start = (input_ids[0] == self.tokenizer.convert_tokens_to_ids("<|Assistant|>")).nonzero(as_tuple=True)[0][0] + 1
        log_probs_answer_part = log_probs_answer[0, answer_start:]

        total_log_prob = log_probs_answer_part.mean().item()

        return math.exp(total_log_prob)  # = p(answer | question, imgs)

    


def add_gaussian_noise(img, std=0.1):
    transform_to_tensor = T.ToTensor()
    transform_to_pil = T.ToPILImage()

    tensor_img = transform_to_tensor(img)
    noise = torch.randn(tensor_img.size()) * std
    noisy_img = tensor_img + noise
    noisy_img = torch.clamp(noisy_img, 0, 1)

    return transform_to_pil(noisy_img)

if __name__ == "__main__":
    question = "Discribe these images. <image><image><image>"
    img_files = [Image.open(f"test_{i + 1}.jpg").convert("RGB") for i in range(3)]

    lvlm = DeepSeekVL2("deepseek-vl2-tiny")
    answer = lvlm(question, img_files)
    print(answer)

    # Add noise
    std = 0.1  # Bạn có thể thử các giá trị như 0.05, 0.1, 0.2
    noisy_imgs = [add_gaussian_noise(img, std=std) for img in img_files]
    [noisy_img.save(f"test_{i + 1}_noisy.jpg") for i, noisy_img in enumerate(noisy_imgs)]
    adv_answer = lvlm(question, noisy_imgs)
    print(adv_answer)
