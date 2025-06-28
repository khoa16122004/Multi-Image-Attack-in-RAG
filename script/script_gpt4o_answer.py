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
    
    output_dir = f"gpt_visual_rag_result"
    os.makedirs(output_dir, exist_ok=True)

    # model
    loader = ImagesLoader(path=args.annotation_path, img_dir=args.dataset_dir)

    llm = GPTService(model_name="gpt-4o")
    
    system_prompt = (
        "You are a smart assistant to answer the question."
        "Just return your answer and nothing else."
    )

    prompt_template = "Question: {question}"
    
    for sample_id in tqdm(range(len(loader))):    
        question, answer, paths, gt_paths = loader.take_data(sample_id)
        
        llm_answer = llm.text_to_text(
            system_prompt=system_prompt,
            prompt=prompt_template.format(question=question),
        ).strip()
        
        meta_data = {
            "question": question,
            "answer": answer,
            "llm_answer": llm_answer,
        }
        
        # save
        sample_dir = os.path.join(output_dir, str(sample_id))
        os.makedirs(sample_dir, exist_ok=True)
        
        with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
            json.dump(meta_data, f, indent=4)

        
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
