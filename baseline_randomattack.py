import os
import sys
import torch
from PIL import Image
import argparse
from util import DataLoader
from fitness import MultiScore
from algorithm import RandomAttack
from tqdm import tqdm
import json
import pickle


def main(args):
    loader = DataLoader(retri_dir=args.result_clean_dir)
    # fitness
    fitness = MultiScore(reader_name=args.reader_name, 
                         retriever_name=args.retriever_name
                         )
    
    # result_dir
    result_dir = f"baseline_randomattack_usingquestion={args.using_question}"
    os.makedirs(result_dir, exist_ok=True)
    
    # sample_path
    with open(args.sample_path, "r") as f:
        sample_ids = [int(line.strip()) for line in f]
    
    # for i in range(args.start_idx, len(loader)):    
    for i in sample_ids:    
        # take data
        question, answer, query, gt_basenames, retri_basenames, retri_imgs, sims = loader.take_retri_data(i)
        
        # init fitness data
        top_adv_imgs = []
        for k in range(1, args.n_k):
            with open(os.path.join(result_dir, f"{args.retriever_name}_{args.reader_name}_{args.std}", str(i), f"adv_{k}.pkl"), "rb") as f:
                top_adv_imgs.append(pickle.load(f))
        top_original_imgs = retri_imgs[:args.n_k]
        golden_answer = fitness.reader.image_to_text(question, top_original_imgs)[0]     
        fitness.init_data(query, 
                          question, 
                          top_adv_imgs, # top_adv_imgs: I'_0 , I'_1, ..., I'_{nk-2}
                          top_original_imgs,  # top_orginal_imgs: I_0, I_1, ..., I_{nk-1}
                          golden_answer,
                          args.n_k)
        
        # algorithm
        algorithm = RandomAttack(
            w=args.w,
            h=args.h,
            fitness=fitness,
            std=args.std,
            sample_id=str(i),
            log_dir=result_dir,
            n_k=args.n_k
        )

        algorithm.solve()
        # break
        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_path", type=str)
    parser.add_argument("--result_clean_dir", type=str, required=True)
    parser.add_argument("--reader_name", type=str, default="llava")
    parser.add_argument("--retriever_name", type=str, default="clip")
    parser.add_argument("--w", type=int, default=312, help="Width to resize images")
    parser.add_argument("--h", type=int, default=312, help="Height to resize images")
    parser.add_argument("--pop_size", type=int, default=20, help="Population size for NSGA-II")
    parser.add_argument("--n_k", type=int)
    parser.add_argument("--std", type=float, default=0.1, help="Standard deviation for initialization")
    parser.add_argument("--using_question", type=int, default=1) # 1 là sử dụng question để query
    
    args = parser.parse_args()  
    main(args)
