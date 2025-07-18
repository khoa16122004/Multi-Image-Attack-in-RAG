import pickle
import numpy as np
import json
from util import Evaluator, EvaluatorEachScore
from tqdm import tqdm
from llm_service import GPTService
import argparse
from reader import Reader
from retriever import Retriever

def main(args):
    with open(args.sample_path, "r") as f:
        sample_ids = [int(line.strip()) for line in f]   
    
    evaluator = EvaluatorEachScore(args)
    for sample_id in tqdm(sample_ids):
        evaluator.evaluation(sample_id, mode=args.mode)
        # raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_k", type=int, required=True)
    parser.add_argument("--retriever_name", type=str, default= "clip")
    parser.add_argument("--reader_name", type=str, default="llava-one")
    parser.add_argument("--std", type=float, default=0.05)
    parser.add_argument("--run_path", type=str, default="run.txt")
    parser.add_argument("--attack_result_path", type=str)
    parser.add_argument("--result_clean_dir", type=str)
    parser.add_argument("--sample_path", type=str)
    parser.add_argument("--end_to_end_dir", type=str)
    parser.add_argument("--using_question", type=int, default=1)
    parser.add_argument("--method", type=str, default="random", choices=["random", "nsga2", "ga"])
    parser.add_argument("--llm", type=str, choices=['gpt', 'llama', 'gemma'])
    parser.add_argument("--target_answer", type=str, choices=['golden_answer', 'gt_answer'])
    parser.add_argument("--mode", type=str, choices=['all', 'end_to_end', 'retrieval'])
    # if non method in the path, it's nsgaii
    args = parser.parse_args()
    main(args)
    
    # param
    
# CUDA_VISIBLE_DEVICES=3 python run_score_for_each_topk.py --n_k 1 --retriever_name clip --reader_name llava-one --std 0.05 --attack_result_path attack_result_usingquestion\=1/clip_llava-one_0.05 --result_clean_dir result_usingquery\=0_clip --sample_path run.txt --using_question 1 --method nsga2 --llm gpt --target_answer golden_answer
