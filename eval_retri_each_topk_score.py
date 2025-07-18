import os
import json
import numpy as np
from tqdm import tqdm
import argparse

def main(args):
    clean_rate_totals = np.zeros(5, dtype=float)
    clean_mrr_totals = np.zeros(5, dtype=float)

    sample_list = os.listdir(args.extracted_path)
    for sample_id in tqdm(sample_list):
        for k in range(5):
            score_file = os.path.join(
                args.extracted_path, sample_id,
                f"inject_{args.n_k}", f"retrieval_{k+1}.json"
            )
            with open(score_file) as f:
                data = json.load(f)
                clean_rate_totals[k] += data["clean_rate_topk"]
                clean_mrr_totals[k] += data["clean_mrr_topk"]
            


    means_clean_rate = (clean_rate_totals / len(sample_list)).tolist()
    means_mrr = (clean_mrr_totals / len(sample_list)).tolist()
    print("Mean clean_rate per top‑k position:", means_clean_rate)
    print("Mean clean mrrscore per top‑k position:", means_mrr)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_k", type=int, required=True)
    p.add_argument("--extracted_path", type=str, required=True)
    main(p.parse_args())
