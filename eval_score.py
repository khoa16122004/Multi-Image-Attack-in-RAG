import pickle
import numpy as np
import json
from util import DataLoader, arkiv_proccess, greedy_selection, get_prompt_compare_answer, parse_score, get_bertscore, Evaluator
from tqdm import tqdm
from llm_service import GPTService
import argparse
from reader import Reader
from retriever import Retriever

def main(args):
    with open(args.sample_path, "r") as f:
        sample_ids = [int(line.strip()) for line in f]   
    
    evaluator = Evaluator(args)
    for sample_id in sample_ids:
        average_scores, attack_success = evaluator.cal_fitness_score(sample_id)
        print(average_scores, attack_success)
        recall_topk, recall_end_to_end = evaluator.cal_recall_end_to_end(sample_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_k", type=int, required=True)
    # parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--retriever_name", type=str, default="clip")
    parser.add_argument("--reader_name", type=str, default="llava-one")
    parser.add_argument("--std", type=float, default=0.05)
    parser.add_argument("--run_path", type=str, default="run.txt")
    args = parser.parse_args()
    main(args)
    
    # param
