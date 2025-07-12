from PIL import Image
import torch
import torchvision.transforms as T
from mantis.models.mllava import (
    MLlavaProcessor,
    LlavaForConditionalGeneration,
    chat_mllava,
)

class Mantis:
    def __init__(self, pretrained: str):
        self.processor = MLlavaProcessor.from_pretrained(f"TIGER-Lab/{pretrained}")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            f"TIGER-Lab/{pretrained}",
            device_map=f"cuda:{torch.cuda.current_device()}",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.kwargs = dict(max_new_tokens=512, num_beams=1, do_sample=False)

    def __call__(self, prompt: str, images):
        # Ép dtype tay nếu processor không xử lý đúng dtype
        img_inputs = self.processor(images=images, return_tensors="pt")
        img_inputs = {k: v.to(self.model.device, dtype=self.model.dtype) for k, v in img_inputs.items()}

        inputs = self.processor.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            pixel_values=img_inputs["pixel_values"],
            **self.kwargs
        )
        return self.processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    @torch.inference_mode()
    def compute_log_prob(self, prompt: str, images, answer: str) -> float:
        full = f"{prompt.strip()} {answer}"
        tok = self.processor.tokenizer
        ids = tok(full, add_special_tokens=False, return_tensors="pt").input_ids.to(self.model.device)
        ans_ids = tok(answer, add_special_tokens=False).input_ids
        labels = ids.clone()
        labels[:, :-len(ans_ids)] = -100

        img_inputs = self.processor(images=images, return_tensors="pt")
        img_inputs = {k: v.to(self.model.device, dtype=self.model.dtype) for k, v in img_inputs.items()}

        loss = self.model(input_ids=ids, pixel_values=img_inputs["pixel_values"], labels=labels).loss
        return torch.exp(-loss * len(ans_ids)).item()
    
    
if __name__ == "__main__":
    q = "Describe these images. <image><image><image>"
    imgs = [Image.open(f"test_{i+1}.jpg").convert("RGB") for i in range(3)]

    vlm = Mantis("Mantis-llava-7b")
    a_clean = vlm(q, imgs)
    p_clean = vlm.compute_log_prob(q, imgs, a_clean)

    imgs_noisy = [add_gaussian_noise(im, 0.05) for im in imgs]
    for i, img in enumerate(imgs_noisy):
        img.save(f"test_{i+1}_noisy.jpg")

    a_adv = vlm(q, imgs_noisy)
    p_adv = vlm.compute_log_prob(q, imgs_noisy, a_adv)

    print(a_clean)
    print(a_adv)
    print(p_adv, p_clean, p_adv / p_clean)
