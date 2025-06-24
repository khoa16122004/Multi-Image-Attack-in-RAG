import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from retriever import Retriever
from reader import Reader
import numpy as np
import pickle as pkl
from util import arkiv_proccess
from PIL import Image
import sys
from util import DataLoader
from copy import deepcopy
import os
import json

class MultiScore:
    def __init__(self, reader_name, retriever_name):
        self.reader = Reader(reader_name)
        self.retriever = Retriever(retriever_name)
        self.retriever_name = retriever_name
        self.reader_name = reader_name
    
    def init_data(self, query, question, top_adv_imgs, top_orginal_imgs, answer, n_k): # top_adv_imgs not inlucde current
        
        # top_adv_imgs: I'_0 , I'_1, ..., I'_{nk-2}
        # top_orginal_imgs: I_0, I_1, ..., I_{nk-1}
        
        self.reader.init_data(answer)
        
        self.original_img = deepcopy(top_orginal_imgs[-1]) # topk original img
        self.top1_img = deepcopy(top_orginal_imgs[0])
        self.top_adv_imgs = top_adv_imgs
        self.n_k = n_k
        self.original_img_tensor = transforms.ToTensor()(self.original_img).cuda()
        self.retri_clean_reuslt = self.retriever(query, [self.top1_img]) # s(q, I_0)
        self.reader_clean_result = self.reader(question, [top_orginal_imgs]) # p(a | I_nk, q)
        
        self.answer = answer
        self.question = question
        self.query = query
        
    def all_equal(self, perturbations: torch.Tensor) -> bool:
        return (perturbations == perturbations[0]).all().item()
    
    def __call__(self, pertubations):  # pertubations: tensor
        pertubations_ = deepcopy(pertubations)
        adv_img_tensors = pertubations_ + self.original_img_tensor
        adv_img_tensors = adv_img_tensors.clamp(0, 1)
        adv_imgs = [to_pil_image(img_tensor) for img_tensor in adv_img_tensors]

        retrieval_result = self.retriever(self.query, adv_imgs)
        
        # adv_top_nk
        adv_topk_imgs = [self.top_adv_imgs + [adv_img] for adv_img in adv_imgs]

        reader_result = self.reader(self.question, adv_topk_imgs)

        retri_scores = (self.retri_clean_reuslt / retrieval_result).cpu().numpy()
        reader_scores = (reader_result / self.reader_clean_result).cpu().numpy()

        return retri_scores, reader_scores,  adv_imgs  
    
    
if __name__ == "__main__":
    retri_dir  = "retri_result_clip"
    reader_dir = r"reader_result/Llama-7b/clip"
    sample_id = 0
    n_k = 5
    
    loader = DataLoader(retri_dir=retri_dir)
    fitness = MultiScore(reader_name="llava", 
                         retriever_name="clip"
                         )

    
    result_dir = f"attack_result"
    question, answer, query, gt_basenames, retri_basenames, retri_imgs = loader.take_data(sample_id)
    json_path = os.path.join(reader_dir, str(sample_id), "answers.json")
    with open(json_path, "r") as f:
        data = json.load(f)
        golder_answer =  data['topk_results'][f'top_{n_k}']['model_answer']
    tzop_adv_imgs = [Image.open(os.path.join(result_dir, f"clip_llava_0.1", "0", f"adv_{k}.pkl")) for k in range(1, n_k)]
    
    
    print(fitness.retriever(query, retri_imgs))
    print(fitness.retriever(query, top_adv_imgs))


   
    

