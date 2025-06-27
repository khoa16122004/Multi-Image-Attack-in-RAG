import os
import json
import argparse
from tqdm import tqdm
from llm_service import GPTService
from util import get_prompt_compare_answer

def main(args):
    llm = GPTService(model_name="gpt-4o")

    for folder_name in tqdm(os.listdir(args.extracted_path)):
        file_path = os.path.join(args.extracted_path, folder_name, "metadata.json")
        if not os.path.exists(file_path):
            continue

        with open(file_path, "r") as f:
            meta_data = json.load(f)

        question = meta_data.get("question", "")
        gt_answer = meta_data.get("answer", "")
        pred_answer = meta_data.get("llm_answer", meta_data.get("lvlm_answer", ""))

        system_prompt, user_prompt = get_prompt_compare_answer(
            gt_answer=gt_answer,
            model_answer=pred_answer,
            question=question
        )

        score_response = llm.text_to_text(
            system_prompt=system_prompt,
            prompt=user_prompt
        ).strip()

        score_file_path = os.path.join(args.extracted_path, folder_name, "score.txt")
        with open(score_file_path, "w") as f:
            f.write(score_response + "\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
