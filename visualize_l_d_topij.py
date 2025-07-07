from util import *
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse


def main(args):
    
    model = [
        "llava-one",
        "llava-next",
        # "deepseek-vl2-tiny",
        # "qwen-vl",
        # "itern-vl"
    ]

    color = [
        'blue',
        'red'
    ]

    lines = [int(line.strip()) for line in open(args.sample_path, "r")]

    figure, ax = plt.subplots(3, 5) # n_k x n_model

    for idx, model_name in enumerate(model):
        model_result_dir = os.path.join(args.attack_result_dir, f"clip_{model_name}_{args.std}")
        full_score_0 = []
        full_score_1 = []
        for sample_id in lines:
            path = os.path.join(model_result_dir, str(sample_id), f"scores_{args.n_k}.pkl")
            with open(path, "rb") as f:
                scores = pickle.load(f)
                scores = arkiv_proccess(scores)
                for i, gen in enumerate(scores):
                    print(i)
                    np.min(gen[:, 0])
                print(np.min(scores[0][:, 0]))
                min_scores_0 = [np.min(gen[:, 0]) for gen in scores]
                min_scores_1 = [np.min(gen[:, 1]) for gen in scores]
                full_score_0.append(min_scores_0)
                full_score_1.append(min_scores_1)

        full_score_0 = np.array(full_score_0)
        full_score_1 = np.array(full_score_1)

        row = 0
        col = idx
        ax[row, col].plot(full_score_0, label='Retrieval Error Score', color=color[0])
        ax[row, col].plot(full_score_1, label='Generation Error Score', color=color[1])
        ax[row, col].set_title(model_name)
        ax[row, col].set_ylim([0, 1])
        ax[row, col].legend()

    plt.tight_layout()
    
    save_path = f"figure_visuattack_plot_std{args.std}_nk{args.n_k}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Figure saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_path", type=str)
    parser.add_argument("--n_k", type=int, default=1, help="Number of attack")
    parser.add_argument("--std", type=float, default=0.1, help="Standard deviation for initialization")
    parser.add_argument("--attack_result_dir", type=str, required=True)    
    args = parser.parse_args()  
    main(args)




