import os
import torch
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from util import arkiv_proccess


def main(args):
    models = [
        "llava-one",
        "llava-next",
        "deepseek-vl2-tiny",
        "qwen-vl",
        "itern-vl"
    ]

    colors = ['blue', 'red']
    model_count = 5
    n_k_list = [1, 2, 3]
    lines = [int(line.strip()) for line in open(args.sample_path, "r")]

    fig, ax = plt.subplots(len(n_k_list), model_count, figsize=(5 * model_count, 4 * len(n_k_list)))

    if len(n_k_list) == 1:
        ax = [ax]
    if model_count == 1:
        ax = [[a] if isinstance(a, plt.Axes) else a for a in ax]

    for row, n_k in enumerate(n_k_list):
        for col in range(model_count):
            model_name = models[col]
            model_result_dir = os.path.join(args.attack_result_dir, f"clip_{model_name}_{args.std}")
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

                ax[row][col].plot(mean_score_0, label='Retrieval Error Score', color=colors[0])
                ax[row][col].plot(mean_score_1, label='Generation Error Score', color=colors[1])
                ax[row][col].legend()

            except FileNotFoundError:
                print(f"⚠️  Missing data for model={model_name}, n_k={n_k}. Leaving blank.")
                ax[row][col].text(0.5, 0.5, "No Data", fontsize=12, ha='center', va='center')
                ax[row][col].set_xticks([])
                ax[row][col].set_yticks([])

            ax[row][col].set_title(f"{model_name} | n_k={n_k}")
            ax[row][col].set_ylim([0.5, 1.2])
            ax[row][col].set_xlabel("Generation Step")
            ax[row][col].set_ylabel("Min Error Score")

    plt.tight_layout()
    os.makedirs("figure_visuattack_plot_std", exist_ok=True)
    save_path = f"figure_visuattack_plot_std/std{args.std}_nk1-2-3.pdf"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Figure saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_path", type=str, required=True)
    parser.add_argument("--std", type=float, default=0.1, help="Standard deviation for initialization")
    parser.add_argument("--attack_result_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)
