import pickle
import numpy as np
import json
from util import DataLoader, arkiv_proccess, greedy_selection
n_k = 1
retriever_name = "clip"
reader_name = "llava-one"
std = 0.05
sample_id = 0

# calculate the L_topi and D_topi

scores_path = f"attack_result/{retriever_name}_{reader_name}_{std}/{sample_id}/scores_{n_k}.pkl"
with open(scores_path, "rb") as f:
    scores = pickle.load(f)
    scores = arkiv_proccess(scores)

    final_front_score = scores[-1]
    print("Final score: ", final_front_score)
    selected_idx = greedy_selection(final_front_score)
    selected_score = final_front_score[selected_idx]
    print(selected_score)
    # greedy selection
    
