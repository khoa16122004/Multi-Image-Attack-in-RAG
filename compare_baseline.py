import os
import torch
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from util import arkiv_proccess
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rc('legend', fontsize=25)  # Giảm từ 25 xuống 20
plt.rc('xtick', labelsize=25)  # Giảm từ 25 xuống 20
plt.rc('ytick', labelsize=25)  # Giảm từ 25 xuống 20
plt.rc('axes', labelsize=24)   # Giảm từ 30 xuống 24


def main(args):
    models = {
        "llava-one": "LLaVA-One.",
        "llava-next": "LLaVA-Next.",
        "deepseek-vl2-tiny": "DeepSeekVL2",
        "qwenvl2.5": "Qwen2.5VL"
    }

    n_k = 1
    colors = {'baseline': 'blue', 'nsga-ii': 'red'}
    linestyles = {'baseline': '--', 'nsga-ii': '-'}
    markers = {'baseline': 's', 'nsga-ii': 'o'}
    lines = [int(line.strip()) for line in open(args.sample_path, "r")]

    fig, ax = plt.subplots(2, len(models), figsize=(5 * len(models), 8))

    for col, model_name in enumerate(models):
        # Đọc dữ liệu từ cả hai thư mục
        attack_result_dir = os.path.join(args.attack_result_dir, f"clip_{model_name}_{args.std}")
        baseline_result_dir = os.path.join(args.baseline_reuslt_dir, f"clip_{model_name}_{args.std}")
        
        result_dirs = {
            'nsga-ii': attack_result_dir,
            'baseline': baseline_result_dir
        }
        
        for method_name, result_dir in result_dirs.items():
            full_score_0, full_score_1 = [], []
            
            try:
                for sample_id in lines:
                    path = os.path.join(result_dir, str(sample_id), f"scores_{n_k}.pkl")
                    if not os.path.exists(path):
                        raise FileNotFoundError
                    with open(path, "rb") as f:
                        scores = pickle.load(f)
                        scores = arkiv_proccess(scores)
                        scores = [np.array(gen) for gen in scores]
                        # print(scores[0].shape)
                        print(path)
                        min_scores_0 = [np.min(gen[:, 0]) for gen in scores]
                        min_scores_1 = [np.min(gen[:, 1]) for gen in scores]
                        full_score_0.append(min_scores_0)
                        full_score_1.append(min_scores_1)

                full_score_0 = np.array(full_score_0)
                full_score_1 = np.array(full_score_1)

                mean_score_0 = np.mean(full_score_0, axis=0)
                mean_score_1 = np.mean(full_score_1, axis=0)

                # Chỉ lấy mỗi 5 step
                steps = np.arange(len(mean_score_0))
                sampled_indices = steps[::20]
                ax[0][col].plot(sampled_indices, mean_score_0[sampled_indices], 
                               label=method_name, color=colors[method_name], 
                               linestyle=linestyles[method_name], marker=markers[method_name], 
                               linewidth=2, markersize=6)
                ax[1][col].plot(sampled_indices, mean_score_1[sampled_indices], 
                               label=method_name, color=colors[method_name],
                               linestyle=linestyles[method_name], marker=markers[method_name], 
                               linewidth=2, markersize=6)

            except FileNotFoundError:
                print(f"⚠️  Missing data for model={model_name}, method={method_name}. Skipping.")
                continue

        ax[0][col].set_title(f"{models[model_name]}", fontsize=16)
        ax[1][col].set_title(f"{models[model_name]}", fontsize=16)
        ax[0][col].set_xlabel("Generation Step")
        ax[1][col].set_xlabel("Generation Step")
        ax[0][col].set_ylabel("Retrieval Error score")
        ax[1][col].set_ylabel("Generation Error score")

        # Legend giống như trong hình mẫu
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
    save_path = f"figure_visuattack_plot_std/std{args.std}_baseline_vs_nsgaii.pdf"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Figure saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_path", type=str, required=True)
    parser.add_argument("--std", type=float, default=0.1, help="Standard deviation for initialization")
    parser.add_argument("--attack_result_dir", type=str, required=True)
    parser.add_argument("--baseline_reuslt_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)