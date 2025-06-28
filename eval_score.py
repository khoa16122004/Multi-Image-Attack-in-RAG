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
    with open(args.sample_path, "r") as f:
        sample_ids = [int(line.strip()) for line in f]   
    
    evaluator = Evaluator(args)
    for sample_id in tqdm(sample_ids):
        evaluator.evaluation(sample_id)
        # raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_k", type=int, required=True)
    # parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--retriever_name", type=str, default="clip")
    parser.add_argument("--reader_name", type=str, default="llava-one")
    parser.add_argument("--std", type=float, default=0.05)
    parser.add_argument("--run_path", type=str, default="run.txt")
    parser.add_argument("--attack_result_path", type=str)
    parser.add_argument("--result_clean_dir", type=str)
    parser.add_argument("--sample_path", type=str)
    parser.add_argument("--end_to_end_dir", type=str)
    parser.add_argument("--using_question", type=int, default=1)
    args = parser.parse_args()
    main(args)
    
    # param
