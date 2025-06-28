import os
import pickle
import numpy as np
import json
from util import Evaluator
from tqdm import tqdm
from llm_service import GPTService
import argparse
from reader import Reader
from retriever import Retriever

def main(args):
    fitness_score = []
    attack_success_total = 0
    recall_topk_total = 0
    end_to_end_score_total = 0

    sample_list = os.listdir(args.extracted_path)
    for sample_id in tqdm(sample_list):
        sample_path = os.path.join(args.extracted_path, sample_id) 
        score_path = os.path.join(sample_path, f"scores_{args.n_k}.json")  # fixed formatting

        with open(score_path, "r") as f:
            scores = json.load(f)
        
        fitness_scores = scores["fitness_scores"]
        attack_success = scores["attack_success"]
        recall_topk = scores["recall_topk"]
        end_to_end_score = scores["recall_end_to_end"]
        
        fitness_score.append(fitness_scores)
        attack_success_total += attack_success
        recall_topk_total += recall_topk
        end_to_end_score_total += end_to_end_score
    
    fitness_score = np.array(fitness_score)

    num_samples = len(sample_list)
    print(f"Fitness Score (mean): {fitness_score.mean():.4f}")
    print(f"Attack Success Rate: {attack_success_total / num_samples:.4f}")
    print(f"Recall@Top-{args.n_k}: {recall_topk_total / num_samples:.4f}")
    print(f"End-to-End@Top-{args.n_k}: {end_to_end_score_total / num_samples:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_k", type=int, required=True)
    parser.add_argument("--extracted_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
