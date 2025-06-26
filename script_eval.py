import pickle
import numpy as np
import json
from util import DataLoader, arkiv_proccess, greedy_selection, get_prompt_compare_answer, parse_score, get_bertscore
from tqdm import tqdm
from llm_service import GPTService
import argparse
from reader import Reader
from retriever import Retriever

def main(args):
    # variable
    n_k = args.n_k
    retriever_name = args.retriever_name
    reader_name = args.reader_name
    std = args.std
    run_path = args.run_path

    # model
    loader = DataLoader(retri_dir=f"result_{retriever_name}")
    reader = Reader(reader_name)
    retriever = Retriever(retriever_name)

    # data
    all_scores = [] # all scores of L_topi, D_topi
    success_retri_score = 0 # retri success at top-i
    end_to_end_assumption_scores = 0
    end_to_end_assumption_bertscores = 0
    end_to_end_scores = 0
    end_to_end_bertscores = 0
    attack_success = 0


    # run_path
    with open(run_path, "r") as f:
        sample_ids = [int(line.strip()) for line in f]
        
        
    # llm evaluation
    llm = GPTService(model_name="gpt-4o")

    for sample_id in tqdm(sample_ids):
        
        # # take data
        # # # # scores
        # scores_path = f"attack_result/{retriever_name}_{reader_name}_{std}/{sample_id}/scores_{n_k}.pkl"
        # with open(scores_path, "rb") as f:
        #     scores = pickle.load(f)
        #     scores = arkiv_proccess(scores) # #
            
        # # # # metadata
        # metadata_path = f"result_{retriever_name}/{sample_id}/metadata.json"    
        # with open(metadata_path, "r") as f:
        #     meta_data = json.load(f)
        #     question = meta_data["question"] # #
        
        # # # result answer 
        result_path = f"attack_result/{retriever_name}_{reader_name}_{std}/{sample_id}/answers_{n_k}.json"
        with open(result_path, "r") as f:
            data = json.load(f)
            golden_answer = data["golden_answer"]
            adv_answer = data["adv_answer"] # #

        # # caclulate score L_topi , D_topo
        # final_front_score = np.array(scores[-1])
        # selected_scores, success_retri = greedy_selection(final_front_score)
        # all_scores.append(selected_scores)
        # if success_retri == True:
        #     if selected_scores[1] < 1:
        #         attack_success += 1
        #     success_retri_score += 1
        
        # ##########################################################  
        # # Assumption score        
        # # # End-To-End-performance with assumption that top-{i-1} is successfulled pooled        
        # system_prompt, user_prompt = get_prompt_compare_answer(golden_answer, adv_answer, question)
        # llm_output = llm.text_to_text(
        #     system_prompt=system_prompt,
        #     prompt=user_prompt,
        # ).strip()
        # score = parse_score(llm_output)
        # end_to_end_assumption_scores += score
        
        # # # BertScore-End-To-End-performance with assumption that top-{i-1} is successfulled pooled        
        # score = get_bertscore(golden_answer, adv_answer)
        # end_to_end_assumption_bertscores += score
        
        # #######################################################################
        # Real performance
        metadata_path = f"result_{retriever_name}/{sample_id}/metadata.json"    
        with open(metadata_path, "r") as f:
            meta_data = json.load(f)
            question = meta_data["question"] # #
            clean_sims = meta_data["sims"] # # [0.3057, 0.3057]
        question, answer, query, gt_basenames, retri_basenames, clean_imgs = loader.take_retri_data(sample_id)
        
        top_adv_imgs = []
        for k in range(1, n_k + 1):
            adv_img_paths = f"attack_result/{retriever_name}_{reader_name}_{std}/{sample_id}/adv_{k}.pkl"
            with open(adv_img_paths, "rb") as f:
                top_adv_imgs.append(pickle.load(f))
        
        adv_sims = retriever(query, top_adv_imgs).cpu().tolist() # tensor([[0.3057], [0.3057]], device='cuda:0', dtype=torch.float16)
        adv_sims = [item[0] for item in adv_sims]
        all_imgs = clean_imgs + top_adv_imgs
        all_sims = clean_sims + adv_sims
        sorted_indices = sorted(range(len(all_sims)), key=lambda i: all_sims[i], reverse=True)
        top_k_imgs = [all_imgs[i] for i in sorted_indices[:n_k]]
        top_k_sims = [all_sims[i] for i in sorted_indices[:n_k]]
        print(all_sims)
        print(top_k_sims)
        raise
        adv_answer = reader.image_to_text(question, topk_imgs)[0]
        
        ## end-to-end performance score
        system_prompt, user_prompt = get_prompt_compare_answer(golden_answer, adv_answer, question)
        llm_output = llm.text_to_text(
            system_prompt=system_prompt,
            prompt=user_prompt,
        ).strip()
        score = parse_score(llm_output)
        end_to_end_scores += score

        # # End-To-End-performance-bertscores
        score = get_bertscore(golden_answer, adv_answer)
        end_to_end_bertscores += score
        
        
    all_scores = np.array(all_scores)
    average_scores = np.mean(all_scores, axis=0)
    average_success_retri = success_retri_score / len(sample_ids)
    average_end_to_end_assumption_scores = end_to_end_assumption_scores / len(sample_ids)
    average_end_to_end_assumption_bertscores = end_to_end_assumption_bertscores / len(sample_ids)
    attack_success_rate = attack_success / len(sample_ids)

    print("Scores: ", average_scores)    
    print("Success Retri: ", average_success_retri)
    print("Attack Success Rate: ", attack_success_rate)
    print("End-to-End (Assumption) GPT-scores: ", average_end_to_end_assumption_scores)
    print("End-to-End (Assumption) Bertscores: ", average_end_to_end_assumption_bertscores)



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
