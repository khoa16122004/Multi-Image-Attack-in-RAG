import os
import argparse
from tqdm import tqdm
import re

def parse_score(text):
    match = re.search(r"Score:\s*([01](?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    return None

def main(args):
    total_score = 0
    count = 0

    for folder_name in tqdm(os.listdir(args.extracted_path)):
        if not args.n_k:
            file_path = os.path.join(args.extracted_path, folder_name, "score.txt")
        else:
            file_path = os.path.join(args.extracted_path, folder_name, f"score_{args.n_k}.txt")


        score_response = open(file_path, "r").read().strip()
        score = parse_score(score_response)

        if score is not None:
            print(score)
            if score == 0.5:
                score = 1
            total_score += score
            count += 1

    if count > 0:
        avg_score = total_score / count
        print(f"Average score over {count} samples: {avg_score:.4f}")
    else:
        print("No valid scores found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_k", type=int, default=None)
    parser.add_argument("--extracted_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
