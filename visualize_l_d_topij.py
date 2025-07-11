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
plt.rc('legend', fontsize=18)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=20)


def main(args):
    models = ["llava-one", "llava-next", "deepseek-vl2-tiny", "qwenvl2.5", "itern-vl"]
    n_k_list = [1, 2, 3]
    colors = ['blue', 'green', 'red']
    lines = [int(line.strip()) for line in open(args.sample_path)]

    fig, ax = plt.subplots(2, len(models), figsize=(5 * len(models), 8))

    for col, model_name in enumerate(models):
        model_result_dir = os.path.join(args.attack_result_dir, f"clip_{model_name}_{args.std}")

        for idx_nk, n_k in enumerate(n_k_list):
            full_score_0, full_score_1 = [], []

            for sample_id in lines:
                path = os.path.join(model_result_dir, str(sample_id), f"scores_{n_k}.pkl")
                if not os.path.exists(path):
                    continue
                with open(path, "rb") as f:
                    scores = arkiv_proccess(pickle.load(f))
                    scores = [np.array(gen) for gen in scores]
                    full_score_0.append([np.min(gen[:, 0]) for gen in scores])
                    full_score_1.append([np.min(gen[:, 1]) for gen in scores])

            if not full_score_0:
                print(f"⚠️  No data for model={model_name}, n_k={n_k}.")
                continue

            mean_score_0 = np.mean(full_score_0, axis=0)
            mean_score_1 = np.mean(full_score_1, axis=0)

            ax[0][col].plot(mean_score_0, label=f"top_k={n_k}", color=colors[idx_nk])
            ax[1][col].plot(mean_score_1, label=f"top_k={n_k}", color=colors[idx_nk])

        ax[0][col].set_title(model_name, fontsize=16)
        ax[1][col].set_title(model_name, fontsize=16)
        ax[0][col].set_xlabel("Generation Step")
        ax[1][col].set_xlabel("Generation Step")
        ax[0][col].set_ylabel("Retrieval Error score")
        ax[1][col].set_ylabel("Generation Error score")
        ax[0][col].legend()
        ax[1][col].legend()
        if ax[0][col].lines:
            ax[0][col].set_ylim(0.97, 1.02)
        if ax[1][col].lines:
            all_y_1 = np.concatenate([l.get_ydata() for l in ax[1][col].lines])
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
