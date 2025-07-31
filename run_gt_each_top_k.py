import pickle
import numpy as np
import json
from util import Evaluator, EvaluatorEachScore
from tqdm import tqdm
from llm_service import GPTService
import argparse
from reader import Reader
from retriever import Retriever
import os
if __name__ == "__main__":
    import argparse

    # Thêm parser để nhận tham số từ dòng lệnh
    parser = argparse.ArgumentParser(description="Evaluate original predictions using top-k results.")
    parser.add_argument("--retri_dir", type=str, required=True, help="Path to the retrieval directory.")
    parser.add_argument("--run_file", type=str, required=True, help="Path to the file containing sample IDs.")
    parser.add_argument("--retriever_name", type=str, default="clip", help="Name of the retriever.")
    parser.add_argument("--reader_name", type=str, default="llava-one", help="Name of the reader.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top-k images to use for evaluation.")
    args = parser.parse_args()

    output_dir = f"clean_result_{args.retriever_name}_{args.reader_name}_top{args.top_k}"
    # Khởi tạo các đối tượng cần thiết
    evaluator = EvaluatorEachScore(args)

    # Đọc danh sách sample ID từ file
    with open(args.run_file, "r") as f:
        sample_ids = [int(line.strip()) for line in f]

    # Đánh giá original predictions
    results = evaluator.evaluate_original_predictions(sample_ids, top_k=args.top_k)

    # Lưu kết quả ra file JSON
    output_file = os.path.join(output_dir, f"original_eval_top{args.top_k}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation completed. Results saved to {output_file}.")