import pickle
import numpy as np
import json
from util import DataLoader, arkiv_proccess, greedy_selection, get_prompt_compare_answer, parse_score, get_bertscore
from tqdm import tqdm
from llm_service import GPTService

# param
n_k = 1
retriever_name = "clip"
reader_name = "llava-one"
std = 0.05
run_path = "run.txt"

# data
all_scores = [] # all scores of L_topi, D_topi
success_retri_score = 0 # retri success at top-i
end_to_end_assumption_scores = 0
end_to_end_assumption_bertscores = 0
attack_success = 0


# run_path
with open(run_path, "r") as f:
    sample_ids = [int(line.strip()) for line in f]
    
    
# llm evaluation
llm = GPTService(model_name="gpt-4o")

for sample_id in tqdm(sample_ids):
    
    # take data
    # # # scores
    scores_path = f"attack_result/{retriever_name}_{reader_name}_{std}/{sample_id}/scores_{n_k}.pkl"
    with open(scores_path, "rb") as f:
        scores = pickle.load(f)
        scores = arkiv_proccess(scores) # #
        
    # # # metadata
    metadata_path = f"result_{retriever_name}/{sample_id}/metadata.json"    
    with open(metadata_path, "r") as f:
        meta_data = json.load(f)
        question = meta_data["question"] # #
    
    # # # result answer 
    result_path = f"attack_result/{retriever_name}_{reader_name}_{std}/{sample_id}/answers_{n_k}.json"
    with open(result_path, "r") as f:
        data = json.load(f)
        golden_answer = data["golden_answer"]
        adv_answer = data["adv_answer"] # #

    # caclulate score L_topi , D_topo
    final_front_score = np.array(scores[-1])
    selected_scores, success_retri = greedy_selection(final_front_score)
    all_scores.append(selected_scores)
    if success_retri == True:
        if selected_scores[1] < 1:
            attack_success += 1
        success_retri_score += 1
            
    # End-To-End-performance with assumption that top-{i-1} is successfulled pooled        
    system_prompt, user_prompt = get_prompt_compare_answer(golden_answer, adv_answer, question)
    llm_output = llm.text_to_text(
        system_prompt=system_prompt,
        prompt=user_prompt,
    ).strip()
    score = parse_score(llm_output)
    end_to_end_assumption_scores += score
    
    # BertScore-End-To-End-performance with assumption that top-{i-1} is successfulled pooled        
    score = get_bertscore(golden_answer, adv_answer)
    end_to_end_assumption_bertscores += score
    
all_scores = np.array(all_scores)
average_scores = np.mean(all_scores, axis=0)
average_success_retri = success_retri_score / len(sample_ids)
average_end_to_end_assumption_scores = end_to_end_assumption_scores / len(sample_ids)
average_end_to_end_assumption_bertscores = end_to_end_assumption_bertscores / len(sample_ids)
attack_success_rate = attack_success / len(sample_ids)

print("Scores: ", average_scores)    
print("Success Retri: ", average_success_retri)
print("Attack Success Rate: ", attack_success_rate)
print("End-to-End (Assumption scores): ", average_end_to_end_assumption_scores)
print("End-to-End (Assumption scores): ", average_end_to_end_assumption_bertscores)
