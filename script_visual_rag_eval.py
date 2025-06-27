import os
import sys
import torch
from PIL import Image
import argparse
import json
from util import DataLoader, ImagesLoader
from retriever import Retriever
from reader import Reader
from tqdm import tqdm
from llm_service import LlamaService, GPTService
import pickle 
def main(args):
    
    output_dir = f"visual_rag_eval"
    os.makedirs(output_dir, exist_ok=True)

    # model
    loader = DataLoader(img_dir=args.dataset_dir)
    reader = Reader(model_name=args.reader_name)
    
    for sample_id in tqdm(range(len(loader))):    
        question, answer, query, gt_basenames, retri_basenames, retri_imgs = loader.take_retri_data(i)
        
        lvlm_answers = []
        for i in range(3):
            lvlm_answer = reader.image_to_text(question, retri_imgs)[0]
            lvlm_answers.append(lvlm_answer)
        
        meta_data = {
            "question": question,
            "answer": answer,
            "lvlm_answers": lvlm_answers,
        }
        
        # save
        sample_dir = os.path.join(output_dir, str(sample_id))
        os.makedirs(sample_dir, exist_ok=True)
        
        with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
            json.dump(meta_data, f, indent=4)

        
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
