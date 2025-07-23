# import pickle
# import numpy as np
# import json
# from util import Evaluator, EvaluatorEachScore
# from tqdm import tqdm
# from llm_service import GPTService
# import argparse
# from reader import Reader
# from retriever import Retriever

# def main(args):
#     with open(args.sample_path, "r") as f:
#         sample_ids = [int(line.strip()) for line in f]   
    
#     evaluator = EvaluatorEachScore(args)
#     for sample_id in tqdm(sample_ids):
#         evaluator.evaluation(sample_id, mode=args.mode)
#         # raise


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--n_k", type=int, required=True)
#     parser.add_argument("--retriever_name", type=str, default= "clip")
#     parser.add_argument("--reader_name", type=str, default="llava-one")
#     parser.add_argument("--std", type=float, default=0.05)
#     parser.add_argument("--run_path", type=str, default="run.txt")
#     parser.add_argument("--attack_result_path", type=str)
#     parser.add_argument("--result_clean_dir", type=str)
#     parser.add_argument("--sample_path", type=str)
#     parser.add_argument("--end_to_end_dir", type=str)
#     parser.add_argument("--using_question", type=int, default=1)
#     parser.add_argument("--method", type=str, default="random", choices=["random", "nsga2", "ga"])
#     parser.add_argument("--llm", type=str, choices=['gpt', 'llama', 'gemma'])
#     parser.add_argument("--target_answer", type=str, choices=['golden_answer', 'gt_answer'])
#     parser.add_argument("--mode", type=str, choices=['all', 'end_to_end', 'retrieval'])
#     # if non method in the path, it's nsgaii
#     args = parser.parse_args()
#     main(args)
    
#     # param
    
# # CUDA_VISIBLE_DEVICES=3 python run_score_for_each_topk.py --n_k 1 --retriever_name clip --reader_name llava-one --std 0.05 --attack_result_path attack_result_usingquestion\=1/clip_llava-one_0.05 --result_clean_dir result_usingquery\=0_clip --sample_path run.txt --using_question 1 --method nsga2 --llm gpt --target_answer golden_answer



import os
import torch
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from util import arkiv_proccess
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=20)


def main(args):
    models = {
        "llava-one": "LLaVA-One.",
        "llava-next": "LLaVA-Next.",
        "deepseek-vl2-tiny": "DeepSeekVL2",
        "qwenvl2.5": "Qwen2.5VL"
    }

    n_k_list = [1, 2, 3, 4, 5]
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    linestyles = ['-', '--', '-', '-', '-']
    markers = ['s', '^', 'o', 'd', 'v']
    lines = [int(line.strip()) for line in open(args.sample_path, "r")]

    fig, ax = plt.subplots(2, len(models), figsize=(5 * len(models), 8))

    for col, model_name in enumerate(models):
        model_result_dir = os.path.join(args.attack_result_dir, f"clip_{model_name}_{args.std}")
        for idx_nk, n_k in enumerate(n_k_list):
            full_score_0, full_score_1 = [], []

            try:
                for sample_id in lines:
                    path = os.path.join(model_result_dir, str(sample_id), f"scores_{n_k}.pkl")
                    if not os.path.exists(path):
                        raise FileNotFoundError
                    with open(path, "rb") as f:
                        scores = pickle.load(f)
                        scores = arkiv_proccess(scores)
                        scores = [np.array(gen) for gen in scores]
                        min_scores_0 = [np.min(gen[:, 0]) for gen in scores]
                        min_scores_1 = [np.min(gen[:, 1]) for gen in scores]
                        full_score_0.append(min_scores_0)
                        full_score_1.append(min_scores_1)

                full_score_0 = np.array(full_score_0)
                full_score_1 = np.array(full_score_1)

                mean_score_0 = np.mean(full_score_0, axis=0)
                mean_score_1 = np.mean(full_score_1, axis=0)

                # Chỉ lấy mỗi 20 step
                steps = np.arange(len(mean_score_0))
                sampled_indices = steps[::20]
                ax[0][col].plot(sampled_indices, mean_score_0[sampled_indices], 
                               label=f"top_k={n_k}", color=colors[idx_nk],
                               linestyle=linestyles[idx_nk], marker=markers[idx_nk],
                               linewidth=2, markersize=6)
                ax[1][col].plot(sampled_indices, mean_score_1[sampled_indices], 
                               label=f"top_k={n_k}", color=colors[idx_nk],
                               linestyle=linestyles[idx_nk], marker=markers[idx_nk],
                               linewidth=2, markersize=6)

            except FileNotFoundError:
                print(f"⚠️  Missing data for model={model_name}, n_k={n_k}. Skipping.")
                continue

        ax[0][col].set_title(f"{models[model_name]}", fontsize=16)
        ax[1][col].set_title(f"{models[model_name]}", fontsize=16)
        ax[0][col].set_xlabel("Generation Step")
        ax[1][col].set_xlabel("Generation Step")
        ax[0][col].set_ylabel("Retrieval Error score")
        ax[1][col].set_ylabel("Generation Error score")

        # Legend đẹp giống như trong hình mẫu
        ax[0][col].legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax[1][col].legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

        # Auto y-lim
        if len(ax[0][col].lines) > 0:
            all_y_0 = np.concatenate([line.get_ydata() for line in ax[0][col].lines])
            ax[0][col].set_ylim(0.97, 1.02)

        if len(ax[1][col].lines) > 0:
            all_y_1 = np.concatenate([line.get_ydata() for line in ax[1][col].lines])
            ymin_1, ymax_1 = all_y_1.min() - 0.01, all_y_1.max() + 0.1
            ax[1][col].set_ylim(max(ymin_1, 0), min(ymax_1, 1))

    plt.tight_layout()
    os.makedirs("figure_visuattack_plot_std", exist_ok=True)
    save_path = f"figure_visuattack_plot_std/std{args.std}_by_metric.pdf"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Figure saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_path", type=str, required=True)
    parser.add_argument("--std", type=float, default=0.1, help="Standard deviation for initialization")
    parser.add_argument("--attack_result_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)