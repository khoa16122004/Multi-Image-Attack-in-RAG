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
    output_dir = f"result_{args.retriever_name}"
    os.makedirs(output_dir, exist_ok=True)

    loader = ImagesLoader(path=args.annotation_path, img_dir=args.dataset_dir)
    retriever = Retriever(model_name=args.retriever_name)

    llm = GPTService(model_name="gpt-4o")
    
    system_prompt = (
        "You are a smart assistant. Your task is to extract the most useful physical or behavioral feature mentioned in the question "
        "that can help retrieve the most relevant images. These should be short, descriptive, and specific terms such as 'tail', 'claws', 'antennae', "
        "'wings spread', 'flying', 'diving', or 'open mouth'. \n\n"
        "Avoid including species names, color adjectives, vague descriptions, or stop words. Focus only on concrete **physical traits** or **behavioral states** "
        "that are visually observable and useful for filtering images.\n\n"
        "If the question refers to a particular action or state (e.g., 'underwing pattern when birds are flying'), then return the relevant state or action (e.g., 'flying').\n\n"
        "Only return **one concise phrase** or keyword – the most useful one – and nothing else."
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
    parser.add_argument("--retriever_name", type=str, default="clip")
    parser.add_argument("--w", type=int, default=312)
    parser.add_argument("--h", type=int, default=312)
    args = parser.parse_args()
    main(args)
