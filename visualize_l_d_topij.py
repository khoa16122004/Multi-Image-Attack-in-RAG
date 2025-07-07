from util import *
import torch
import matplotlib.pyplot as plt
import numpy as np


model = [
    "llava-one",
    "llava-next",
    "deepseek-vl2-tiny",
    "qwen-vl",
    "itern-vl"
]

color = [
    'blue',
    'red'
]

n_k = 1
attack_result_dir = r"/data/elo/khoatn/VisualRAG/Multi-Image-Attack-in-RAG/attack_result_usingquestion=1"
std="0.05"
run_path = "run.txt"
lines = [int(line.strip()) for line in open(run_path, "r")]
for model_name in model:
    model_result_dir = os.path.join(attack_result_dir, f"{model_name}_{std}", )
    avg_score_0 = []
    avg_score_1 = []
    for sample_id in lines:
        path = os.path.join(model_result_dir, str(sample_id), f"scores_{n_k}.pkl")
        with open(path, "rb") as f:
            scores = pickle.load(f)
            scores = arkiv_proccess(scores)
            print(scores.shape)
            raise
            min_score_0 = np.min(final_front_score[:, 0])
            min_score_1 = np.min(final_front_score[:, 1])
            avg_score_0.append(min_score_0)
            avg_score_1.append(min_score_1)

    # calculate the




