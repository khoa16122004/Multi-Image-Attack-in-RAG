from util import DataLoader, get_prompt_compare_answer, parse_score
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
os.makedirs(output_dir, exist_ok=True)
loader = DataLoader(
    retri_dir=r"result_usingquery=0_clip",
)
# reader = Reader(reader_name="llava-one")

attack_result_dir = f"attack_result_usingquestion=1/clip_{model_name}_0.05"
model_dir = os.path.join(output_dir, f"{model_name}")
for sample_id in sample_ids:
    sample_dir = os.path.join(model_dir, str(sample_id))
    os.makedirs(sample_dir, exist_ok=True)
    for k in range(1, max_topk + 1):
        top_k_path = os.path.join(sample_dir, f"eval_results_top{k}.json")
        question, gt_answer, query, gt_basenames, retri_basenames, retri_imgs, sims = loader.take_retri_data(sample_id)
        with open(os.path.join(attack_result_dir, str(sample_id), f"answers_{k}.json"), "r") as f:
            data = json.load(f)
            golden_answer = data['golden_answer']
        system_prompt, user_prompt = get_prompt_compare_answer(gt_answer=gt_answer, model_answer=golden_answer, question=question)
        score_response = llm.text_to_text(system_prompt=system_prompt, prompt=user_prompt).strip()
        end_to_end_score = parse_score(score_response)        

        output_data = {
            "gt_answer": gt_answer,
            "golden_answer": golden_answer,
            "score_response": score_response,
            "parse_score": end_to_end_score,
        }
        
        # Save the output data to a JSON file
        output_file_path = os.path.join(sample_dir, f"answers_top{k}.json")
        with open(output_file_path, "w") as output_file:
            json.dump(output_data, output_file, indent=4)
        
        raise

