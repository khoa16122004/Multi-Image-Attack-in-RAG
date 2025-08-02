from util import DataLoader, get_prompt_compare_answer, parse_score
from llm_service import GPTService
from reader import Reader
import os
import json
from tqdm import tqdm
from tqdm import tqdm

max_topk = 5
model_name = "deepseek-vl2-tiny"
clean_dir = f"clean_result/{model_name}"
run_path = "run.txt"
with open(run_path, "r") as f:
    sample_ids = [int(line.strip()) for line in f]   

scores_topk = [0, 0, 0, 0, 0]
for k in tqdm(range(1, max_topk + 1)):
    for i in sample_ids:
        answer_path = os.path.join(clean_dir, str(i), f"answers_top{k}.json")
        if not os.path.exists(answer_path):
            raise
        with open(answer_path, "r") as f:
            data = json.load(f)
            score = data['parse_score']
            scores_topk[k - 1] += score
            

print(scores_topk)   