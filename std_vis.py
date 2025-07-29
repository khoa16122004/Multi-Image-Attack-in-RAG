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
plt.rc('legend', fontsize=20)
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
    colors = ['blue', 'green', 'red', 'orange']
    linestyles = ['-', '--', '-', '-']
    markers = ['s', '^', 'o', 'd']
    lines = [int(line.strip()) for line in open(args.sample_path, "r")]

    # Tăng figsize để có chỗ cho legend ở dưới
    fig, ax = plt.subplots(2, len(models), figsize=(5 * len(models), 10))
    
    # Đảm bảo ax luôn là array 2D
    if len(models) == 1:
        ax = ax.reshape(2, 1)

    for col, model_name in enumerate(models):
        model_result_dir = os.path.join(args.attack_result_dir, f"clip_{model_name}_{args.std}")
        
        # Lưu tất cả mean scores của các top_k để tính trung bình và std
        all_mean_scores_0 = []  # shape: (n_k, n_steps)
        all_mean_scores_1 = []  # shape: (n_k, n_steps)
        
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

                # Tính mean cho từng top_k
                mean_score_0 = np.mean(full_score_0, axis=0)
                mean_score_1 = np.mean(full_score_1, axis=0)
                
                all_mean_scores_0.append(mean_score_0)
                all_mean_scores_1.append(mean_score_1)

            except FileNotFoundError:
                print(f"⚠️  Missing data for model={model_name}, n_k={n_k}. Skipping.")
                continue

        # Tính trung bình và std của tất cả các top_k
        if all_mean_scores_0:
            all_mean_scores_0 = np.array(all_mean_scores_0)  # shape: (n_k, n_steps)
            all_mean_scores_1 = np.array(all_mean_scores_1)  # shape: (n_k, n_steps)
            
            # Trung bình của các top_k (trung bình theo axis=0 để lấy trung bình của 5 top_k)
            avg_across_topk_0 = np.mean(all_mean_scores_0, axis=0)  # shape: (n_steps,)
            avg_across_topk_1 = np.mean(all_mean_scores_1, axis=0)  # shape: (n_steps,)
            
            # Std của các top_k
            std_across_topk_0 = np.std(all_mean_scores_0, axis=0)   # shape: (n_steps,)
            std_across_topk_1 = np.std(all_mean_scores_1, axis=0)   # shape: (n_steps,)

            # Chỉ lấy mỗi 20 step
            steps = np.arange(len(avg_across_topk_0))
            sampled_indices = steps[::20]
            
            # Plot trung bình của các top_k
            ax[0][col].plot(sampled_indices, avg_across_topk_0[sampled_indices], 
                           label=f"{models[model_name]}", color=colors[col],
                           linestyle=linestyles[col], marker=markers[col],
                           linewidth=2, markersize=6)
            ax[1][col].plot(sampled_indices, avg_across_topk_1[sampled_indices], 
                           label=f"{models[model_name]}", color=colors[col],
                           linestyle=linestyles[col], marker=markers[col],
                           linewidth=2, markersize=6)
            
            # Plot std boundary của các top_k
            ax[0][col].fill_between(sampled_indices, 
                                   avg_across_topk_0[sampled_indices] - std_across_topk_0[sampled_indices],
                                   avg_across_topk_0[sampled_indices] + std_across_topk_0[sampled_indices],
                                   color=colors[col], alpha=0.2)
            ax[1][col].fill_between(sampled_indices, 
                                   avg_across_topk_1[sampled_indices] - std_across_topk_1[sampled_indices],
                                   avg_across_topk_1[sampled_indices] + std_across_topk_1[sampled_indices],
                                   color=colors[col], alpha=0.2)

        ax[0][col].set_title(f"{models[model_name]}", fontsize=16)
        ax[1][col].set_title(f"{models[model_name]}", fontsize=16)
        ax[0][col].set_xlabel("Generation Step")
        ax[1][col].set_xlabel("Generation Step")
        ax[0][col].set_ylabel("Retrieval Error Score")
        ax[1][col].set_ylabel("Generation Error Score")

        # Auto y-lim
        if len(ax[0][col].lines) > 0:
            ax[0][col].set_ylim(0.97, 1.02)

        if len(ax[1][col].lines) > 0:
            all_y_1 = np.concatenate([line.get_ydata() for line in ax[1][col].lines])
            ymin_1, ymax_1 = all_y_1.min() - 0.01, all_y_1.max() + 0.1
            ax[1][col].set_ylim(max(ymin_1, 0), min(ymax_1, 1))

    # Tạo legend chung ở dưới cả figure
    handles, labels = ax[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
               ncol=len(models), frameon=True, fancybox=True, shadow=True, 
               facecolor='white', framealpha=0.9, fontsize=14)

    plt.tight_layout()
    # Thêm space cho legend ở dưới
    plt.subplots_adjust(bottom=0.1)
    
    os.makedirs("figure_visuattack_plot_std", exist_ok=True)
    save_path = f"figure_visuattack_plot_std/std{args.std}_average_topk.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_path", type=str, required=True)
    parser.add_argument("--std", type=float, default=0.1, help="Standard deviation for initialization")
    parser.add_argument("--attack_result_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)