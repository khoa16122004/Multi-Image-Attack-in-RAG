import pickle
import numpy as np
import json
from util import DataLoader, arkiv_proccess, greedy_selection
from tqdm import tqdm

# param
n_k = 1
retriever_name = "clip"
reader_name = "llava-one"
std = 0.05
sample_id = 0
run_path = "run.txt"

# data
all_scores = []
success_retri_score = 0

# run_path
with open(run_path, "r") as f:
    sample_ids = [int(line.strip()) for line in f]
    
for sample_id in tqdm(sample_ids):
    # calculate the L_topi and D_topi
    scores_path = f"attack_result/{retriever_name}_{reader_name}_{std}/{sample_id}/scores_{n_k}.pkl"
    with open(scores_path, "rb") as f:
        scores = pickle.load(f)
        scores = arkiv_proccess(scores)

        final_front_score = np.array(scores[-1])
        selected_scores, success_retri = greedy_selection(final_front_score)
        all_scores.append(selected_scores)

        if success_retri == True:
            success_retri_score += 1

all_scores = np.array(all_scores)
average_scores = np.mean(all_scores, axis=0)
average_success_retri = success_retri_score / len(sample_ids)
print(average_scores)    
print(average_success_retri)
