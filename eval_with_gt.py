import os
import json
from util import DataLoader, get_prompt_compare_answer, parse_score
from reader import Reader
from retriever import Retriever

def eval_with_gt(sample_ids, retriever, reader, loader, top_k=5):
    """
    Evaluate predictions against ground truth (GT) without any attack.
    
    Args:
        sample_ids (list): Danh sách các sample ID cần đánh giá.
        retriever (Retriever): Đối tượng retriever để lấy thông tin.
        reader (Reader): Đối tượng reader để tạo dự đoán từ hình ảnh.
        loader (DataLoader): Đối tượng loader để lấy dữ liệu mẫu.
        top_k (int): Số lượng top-k hình ảnh để sử dụng cho dự đoán.
    
    Returns:
        list: Danh sách kết quả đánh giá cho từng sample.
    """
    results = []

    for sample_id in sample_ids:
        # Lấy dữ liệu từ sample
        question, gt_answer, query, gt_basenames, retri_basenames, retri_imgs, sims = loader.take_retri_data(sample_id)

        # Dự đoán từ retriever
        pred_answer = reader.image_to_text(question, retri_imgs[:top_k])[0]  # Sử dụng top-k hình ảnh để dự đoán

        # So sánh dự đoán với GT
        system_prompt, user_prompt = get_prompt_compare_answer(
            gt_answer=gt_answer, model_answer=pred_answer, question=question
        )
        score_response = retriever.llm.text_to_text(system_prompt=system_prompt, prompt=user_prompt).strip()
        score = parse_score(score_response)

        # Lưu kết quả
        result = {
            "sample_id": sample_id,
            "question": question,
            "ground_truth": gt_answer,
            "predicted_answer": pred_answer,
            "response_score": score_response,
            "parsed_score": score,
        }
        results.append(result)

    return results


if __name__ == "__main__":
    import argparse

    # Thêm parser để nhận tham số từ dòng lệnh
    parser = argparse.ArgumentParser(description="Evaluate predictions against ground truth (GT) without any attack.")
    parser.add_argument("--retri_dir", type=str, required=True, help="Path to the retrieval directory.")
    parser.add_argument("--run_file", type=str, required=True, help="Path to the file containing sample IDs.")
    parser.add_argument("--output_dir", type=str, default=".", help="Base directory to save evaluation results.")
    parser.add_argument("--retriever_name", type=str, default="clip", help="Name of the retriever.")
    parser.add_argument("--reader_name", type=str, default="llava-one", help="Name of the reader.")
    args = parser.parse_args()

    # Khởi tạo các đối tượng cần thiết
    retriever = Retriever(args.retriever_name)
    reader = Reader(args.reader_name)
    loader = DataLoader(retri_dir=args.retri_dir)

    # Đọc danh sách sample ID từ file
    with open(args.run_file, "r") as f:
        sample_ids = [int(line.strip()) for line in f]

    # Tạo thư mục lưu kết quả theo mô hình
    model_output_dir = os.path.join(args.output_dir, f"clean_gt_{args.reader_name}")
    os.makedirs(model_output_dir, exist_ok=True)

    # Chạy đánh giá cho từng top-k từ 1 đến 5
    for top_k in range(1, 6):
        print(f"Running evaluation for top-{top_k}...")
        results = eval_with_gt(sample_ids, retriever, reader, loader, top_k=top_k)

        # Lưu kết quả ra file JSON
        output_file = os.path.join(model_output_dir, f"eval_results_top{top_k}.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Evaluation for top-{top_k} completed. Results saved to {output_file}.")