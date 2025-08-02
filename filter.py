from util import DataLoader
from llm_service import GPTService
from reader import Reader
import os
import json

max_topk = 5
model_name = "llava-one"
retri_dir = r"result_usingquery=0_clip"
attack_result_dir = f"attack_result_usingquestion=1/clip_{model_name}"
run_path = "run.txt"
with open(run_path, "r") as f:
    sample_ids = [int(line.strip()) for line in f]   
llm = GPTService("gpt-4o")
output_dir = "clean_result"
# os.makedirs(output_dir, eexist_ok=True)
loader = DataLoader(
    retri_dir=r"result_usingquery=0_clip",
)
# reader = Reader(reader_name="llava-one")

attack_result_dir = f"attack_result_usingquestion=1/clip_{model_name}s_0.05"

for sample_id in sample_ids:
    for k in range(1, max_topk + 1):
        question, gt_answer, query, gt_basenames, retri_basenames, retri_imgs, sims = loader.take_retri_data(sample_id)
        with open(os.path.join(attack_result_dir, str(sample_id), f"answers_{k}.json", "r")) as f:
            data = json.load(f)
            golden_answer = data['golden_answer']
        print(golden_answer)
        print(gt_answer)
        break