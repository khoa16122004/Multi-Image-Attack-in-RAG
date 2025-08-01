import argparse
from util import EvalProcessTableGT

def main():
    # Khởi tạo các tham số cần thiết
    parser = argparse.ArgumentParser(description="Evaluate average scores for samples.")
    parser.add_argument("--reader_name", type=str, required=True, help="Tên của reader model.")
    parser.add_argument("--retriever_name", type=str, required=True, help="Tên của retriever model.")
    parser.add_argument("--std", type=float, required=True, help="Standard deviation.")
    parser.add_argument("--n_k", type=int, required=True, help="Số lượng top-k.")
    parser.add_argument("--attack_result_path", type=str, required=True, help="Đường dẫn tới kết quả tấn công.")
    parser.add_argument("--result_clean_dir", type=str, required=True, help="Đường dẫn tới thư mục dữ liệu sạch.")
    parser.add_argument("--llm", type=str, required=True, choices=["llama", "gpt"], help="Tên của mô hình LLM.")
    parser.add_argument("--method", type=str, required=True, help="Phương pháp được sử dụng.")
    parser.add_argument("--target_answer", type=str, required=True, help="Loại câu trả lời mục tiêu.")
    parser.add_argument("--run_file", type=int, nargs="+", required=True, help="Danh sách các sample ID.")

    args = parser.parse_args()
    with open(args.run_file, "r") as f:
        sample_ids = [int(line.strip()) for line in f]
    # Khởi tạo evaluator
    evaluator = EvalProcessTableGT(args)

    # Tính trung bình các mẫu dựa trên score
    average_scores = evaluator.calculate_average_scores(sample_ids, args.n_k)
    print("Average Scores:", average_scores)

if __name__ == "__main__":
    main()