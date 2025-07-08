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
            {
                "role": "<|User|>",
                "content": question,
            },
            {"role": "<|Assistant|>", "content": ""}
        ]
        
        prepare_inputs = self.vl_chat_proccessor(
            conversations=conversation,
            images=img_files,
            force_batchify=True,
            system_prompt=""
        ).to(self.vl_gpt.device)
        
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False, return_tensors="pt").to(self.vl_gpt.device)
        
        input_with_answer = torch.cat([prepare_inputs.input_ids, answer_ids], dim=1)
        
        labels = input_with_answer.clone()
        labels[0, :prepare_inputs.input_ids.shape[1]] = -100
        
        attention_mask = torch.ones_like(input_with_answer)
        attention_mask[0, :prepare_inputs.attention_mask.shape[1]] = prepare_inputs.attention_mask[0]
        
        # Extend images_seq_mask cho answer tokens
        if hasattr(prepare_inputs, 'images_seq_mask'):
            images_seq_mask = torch.zeros_like(input_with_answer, dtype=torch.bool)
            images_seq_mask[0, :prepare_inputs.images_seq_mask.shape[1]] = prepare_inputs.images_seq_mask[0]
        else:
            images_seq_mask = None
        
        prepare_inputs_full = prepare_inputs
        prepare_inputs_full.input_ids = input_with_answer
        prepare_inputs_full.attention_mask = attention_mask
        if images_seq_mask is not None:
            prepare_inputs_full.images_seq_mask = images_seq_mask
        
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs_full)
        
        with torch.no_grad():
            outputs = self.vl_gpt.language(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            loss = outputs.loss
        
        num_answer_tokens = answer_ids.shape[1]
        total_log_prob = -loss.item() * num_answer_tokens
        prob = math.exp(total_log_prob)
        
        return prob
        


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
    print(lvlm.compute_log_prob(question, img_files, answer[0]))
    print(answer)

    # Add noise
    std = 100  # Bạn có thể thử các giá trị như 0.05, 0.1, 0.2
    noisy_imgs = [add_gaussian_noise(img, std=std) for img in img_files]
    [noisy_img.save(f"test_{i + 1}_noisy.jpg") for i, noisy_img in enumerate(noisy_imgs)]
    adv_answer = lvlm(question, noisy_imgs)
    print(adv_answer)
    print(lvlm.compute_log_prob(question, noisy_imgs, answer[0]))