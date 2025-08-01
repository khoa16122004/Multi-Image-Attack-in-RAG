import os
import json
import numpy as np
from tqdm import tqdm
import argparse

def main(args):
    totals = np.zeros(5, dtype=float)
    with open(args.run_path, "r") as f:
        sample_list = [line.strip() for line in f]  
    
    for sample_id in tqdm(sample_list):
        for k in range(5):
            score_file = os.path.join(
                args.extracted_path, sample_id,
                f"inject_{args.n_k}", f"answers_{k+1}.json"
            )
            try:
                with open(score_file) as f:
                    totals[k] += json.load(f)["parse_score"]
            except:
                print(f"File not found: {score_file}")
                raise
    means = (totals / len(sample_list)).tolist()
    print("Mean parse_score per top‑k position:", means)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_k", type=int, required=True)
    p.add_argument("--extracted_path", type=str, required=True)
    p.add_argument("--run_path", type=str, required=True)
    main(p.parse_args())
