from mantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration
from mantis.models.mllava import chat_mllava
import torchvision.transforms as T
import torch
from PIL import Image

class Mantis:
    def __init__(self, pretrained):
        self.processor = MLlavaProcessor.from_pretrained(
            f"TIGER-Lab/{pretrained}", trust_remote_code=True
        )

        # ✨ Toàn bộ model ở FP16
        self.model = LlavaForConditionalGeneration.from_pretrained(
            f"TIGER-Lab/{pretrained}",
            device_map=f"cuda:{torch.cuda.current_device()}",
            torch_dtype=torch.float16,            # <-- FP16 (Half)
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )

        # ⚠️ Đồng bộ vision‑tower
        self.model.vision_tower.to(dtype=self.model.dtype, device=self.model.device)

        self.generation_kwargs = {
            "max_new_tokens": 512,
            "num_beams": 1,
            "do_sample": False,
        }

    def __call__(self, qs, img_files):
        inputs = self.processor(images=img_files, text=qs, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

        with torch.inference_mode():
            out_ids = self.model.generate(**inputs, **self.generation_kwargs)

        return self.processor.tokenizer.decode(
            out_ids[0], skip_special_tokens=True
        ).strip()


    @torch.inference_mode()
    def compute_log_prob(self, qs, img_files, ans):
        prompt = qs.strip()
        full_text = f"{prompt} {ans}"
        tok = self.processor.tokenizer

        enc = tok(full_text, add_special_tokens=False, return_tensors="pt")
        input_ids = enc.input_ids.to(self.model.device)

        ans_ids = tok(ans, add_special_tokens=False).input_ids
        labels = input_ids.clone()
        labels[:, :-len(ans_ids)] = -100

        img_inputs = self.processor(images=img_files, return_tensors="pt")
        pixel_values = img_inputs.pixel_values.to(self.model.device, dtype=self.model.dtype)

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=labels,
        )
        nll = outputs.loss * len(ans_ids)
        logp = -nll
        return torch.exp(logp).item()

def add_gaussian_noise(img, std=0.1):
    transform_to_tensor = T.ToTensor()
    transform_to_pil = T.ToPILImage()

    tensor_img = transform_to_tensor(img)
    noise = torch.randn(tensor_img.size()) * std
    noisy_img = tensor_img + noise
    noisy_img = torch.clamp(noisy_img, 0, 1)

    return transform_to_pil(noisy_img)
    
if __name__ == "__main__":
    question = "Describe these images. <image><image><image>"  # Fixed typo
    img_files = [Image.open(f"test_{i + 1}.jpg").convert("RGB") for i in range(3)]

    lvlm = Mantis("Mantis-llava-7b")
    answer = lvlm(question, img_files)
    print(answer)
    
    answer_text = answer if isinstance(answer, str) else answer[0]
    p_clean = lvlm.compute_log_prob(question, img_files, answer_text)

    std = 0.05  
    noisy_imgs = [add_gaussian_noise(img, std=std) for img in img_files]
    [noisy_img.save(f"test_{i + 1}_noisy.jpg") for i, noisy_img in enumerate(noisy_imgs)]
    adv_answer = lvlm(question, noisy_imgs)
    print(adv_answer)
    
    adv_answer_text = adv_answer if isinstance(adv_answer, str) else adv_answer[0]
    p_adv = lvlm.compute_log_prob(question, noisy_imgs, adv_answer_text)
    
    print(f"Clean probability: {p_clean}")
    print(f"Adversarial probability: {p_adv}")
    print(f"Ratio (adv/clean): {p_adv / p_clean}")