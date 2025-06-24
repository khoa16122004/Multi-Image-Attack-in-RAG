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
    output_dir = f"result_{args.reader_name}_{args.retriever_name}"
    os.makedirs(output_dir, exist_ok=True)

    loader = DataLoader(path=args.annotation_path, img_dir=args.dataset_dir)
    retriever = Retriever(model_name=args.retriever_name)

    llm = GPTService(model_name="gpt-4o")
    
    system_prompt = (
        "You are a smart assistant. Your job is to extract the specific physical features mentioned in the question "
        "that are most useful for retrieving relevant images. These should be short and descriptive terms like 'tail', 'claws', 'antennae', etc. "
        "Do not include species names, color words, general descriptions, or stop words. "
        "Only return the most useful physical feature for retrieval – nothing else."
    )
    prompt_template = "Question: {question}"
    
    for sample_id in tqdm(range(len(loader))):    
        question, answer, paths, gt_paths = loader.take_data(sample_id)
        path_basenames = [os.path.basename(path) for path in paths]
        gt_basenames = [os.path.basename(path) for path in gt_paths]
        
        # extract keywords
        keyword_query = llm.text_to_text(
            system_prompt=system_prompt,
            prompt=prompt_template.format(question=question),
        ).strip()
        
        # sims retri
        corpus = []
        basename_corpus = []
        for i, path in enumerate(paths):
            try:
                image = Image.open(path).convert('RGB').resize((args.w, args.h))
                basename_corpus.append(path_basenames[i])
                corpus.append(image)
            except:
                continue

        
        # retri
        sims = retriever(keyword_query, corpus).flatten()
        topk_values, topk_indices = torch.topk(sims, 5)
        topk_basenames = [basename_corpus[i] for i in topk_indices]
        topk_imgs = [corpus[i] for i in topk_indices]


        metadata = {
            "question": question,
            "answer": answer,
            "keyword": keyword_query, 
            "gt_basenames": gt_basenames[:5],
            "topk_basenames": topk_basenames,
            "sims": topk_values.cpu().tolist(),
        }
        
        
        # save
        sample_dir = os.path.join(output_dir, str(sample_id))
        os.makedirs(sample_dir, exist_ok=True)
        for img, basename in zip(topk_imgs, topk_basenames):
            img.save(os.path.join(sample_dir, basename))
        
        with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
            
        with open(os.path.join(sample_dir, "retri_images.pkl"), "wb") as f:
            pickle.dump(topk_imgs, f)
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--reader_name", type=str, default="llava")
    parser.add_argument("--retriever_name", type=str, default="clip")
    parser.add_argument("--w", type=int, default=312)
    parser.add_argument("--h", type=int, default=312)
    args = parser.parse_args()
    main(args)
