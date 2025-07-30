import os
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from util import arkiv_proccess


def main(args):
    models = {
        "llava-one": "LLaVA-One",
        "llava-next": "LLaVA-Next",
        "deepseek-vl2-tiny": "DeepSeekVL2",
        "qwenvl2.5": "Qwen2.5VL"
    }

    n_k_list = [1, 2, 3, 4, 5]
    lines = [int(line.strip()) for line in open(args.sample_path, "r")]

    # Dictionary để lưu kết quả cuối cùng
    final_results = {
        'model': [],
        'score_0_top1_mean': [], 'score_0_top1_std': [],
        'score_0_top2_mean': [], 'score_0_top2_std': [],
        'score_0_top3_mean': [], 'score_0_top3_std': [],
        'score_0_top4_mean': [], 'score_0_top4_std': [],
        'score_0_top5_mean': [], 'score_0_top5_std': [],
        'score_1_top1_mean': [], 'score_1_top1_std': [],
        'score_1_top2_mean': [], 'score_1_top2_std': [],
        'score_1_top3_mean': [], 'score_1_top3_std': [],
        'score_1_top4_mean': [], 'score_1_top4_std': [],
        'score_1_top5_mean': [], 'score_1_top5_std': []
    }

    for model_name in models:
        model_result_dir = os.path.join(args.attack_result_dir, f"clip_{model_name}_{args.std}")
        final_results['model'].append(models[model_name])
        
        for n_k in n_k_list:
            final_scores_0, final_scores_1 = [], []

            try:
                for sample_id in lines:
                    path = os.path.join(model_result_dir, str(sample_id), f"scores_{n_k}.pkl")
                    if not os.path.exists(path):
                        print(f"⚠️  Missing file: {path}")
                        continue
                    
                    with open(path, "rb") as f:
                        scores = pickle.load(f)
                        scores = arkiv_proccess(scores)
                        scores = [np.array(gen) for gen in scores]
                        
                        # Lấy min score của generation cuối cùng
                        final_gen_scores = scores[-1]  # Generation cuối cùng
                        min_score_0 = np.min(final_gen_scores[:, 0])
                        min_score_1 = np.min(final_gen_scores[:, 1])
                        
                        final_scores_0.append(min_score_0)
                        final_scores_1.append(min_score_1)

                # Tính mean và std across 100 samples
                if final_scores_0:
                    mean_0 = np.mean(final_scores_0)
                    std_0 = np.std(final_scores_0)
                    mean_1 = np.mean(final_scores_1)
                    std_1 = np.std(final_scores_1)
                else:
                    mean_0 = std_0 = mean_1 = std_1 = np.nan
                    print(f"⚠️  No data found for model={model_name}, n_k={n_k}")

                # Lưu kết quả vào dictionary
                final_results[f'score_0_top{n_k}_mean'].append(mean_0)
                final_results[f'score_0_top{n_k}_std'].append(std_0)
                final_results[f'score_1_top{n_k}_mean'].append(mean_1)
                final_results[f'score_1_top{n_k}_std'].append(std_1)

            except Exception as e:
                print(f"⚠️  Error processing model={model_name}, n_k={n_k}: {e}")
                # Thêm NaN values nếu có lỗi
                final_results[f'score_0_top{n_k}_mean'].append(np.nan)
                final_results[f'score_0_top{n_k}_std'].append(np.nan)
                final_results[f'score_1_top{n_k}_mean'].append(np.nan)
                final_results[f'score_1_top{n_k}_std'].append(np.nan)

    # Tạo DataFrame
    df = pd.DataFrame(final_results)
    
    # Sắp xếp lại columns để có định dạng như yêu cầu
    score_0_cols = [f'score_0_top{i}_mean' for i in range(1, 6)] + [f'score_0_top{i}_std' for i in range(1, 6)]
    score_1_cols = [f'score_1_top{i}_mean' for i in range(1, 6)] + [f'score_1_top{i}_std' for i in range(1, 6)]
    
    # Tạo bảng với format đẹp hơn
    print("\n" + "="*120)
    print("FINAL RESULTS TABLE - MEAN ± STD ACROSS 100 SAMPLES")
    print("="*120)
    
    # Header
    print(f"{'Model':<15}", end="")
    print(f"{'RETRIEVAL ERROR SCORE (Score 0)':^50}", end="")
    print(f"{'GENERATION ERROR SCORE (Score 1)':^50}")
    
    print(f"{'':15}", end="")
    for i in range(1, 6):
        print(f"{'top-' + str(i):>9}", end="")
    print("  ", end="")
    for i in range(1, 6):
        print(f"{'top-' + str(i):>9}", end="")
    print()
    
    print("-" * 120)
    
    # Data rows
    for idx, model in enumerate(df['model']):
        print(f"{model:<15}", end="")
        
        # Score 0 (Retrieval Error)
        for i in range(1, 6):
            mean_val = df[f'score_0_top{i}_mean'].iloc[idx]
            std_val = df[f'score_0_top{i}_std'].iloc[idx]
            if not np.isnan(mean_val):
                print(f"{mean_val:.3f}±{std_val:.3f}"[:9], end="")
            else:
                print(f"{'N/A':>9}", end="")
        
        print("  ", end="")
        
        # Score 1 (Generation Error)
        for i in range(1, 6):
            mean_val = df[f'score_1_top{i}_mean'].iloc[idx]
            std_val = df[f'score_1_top{i}_std'].iloc[idx]
            if not np.isnan(mean_val):
                print(f"{mean_val:.3f}±{std_val:.3f}"[:9], end="")
            else:
                print(f"{'N/A':>9}", end="")
        print()
    
    print("="*120)
    
    # Lưu ra file CSV
    os.makedirs("results_tables", exist_ok=True)
    csv_path = f"results_tables/final_results_std{args.std}.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"✅ Results saved to {csv_path}")
    
    # Lưu ra file Excel với format đẹp hơn
    excel_path = f"final_results_std0.05.csv/final_results_std{args.std}.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Final_Results', index=False)
    print(f"✅ Results saved to {excel_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_path", type=str, required=True)
    parser.add_argument("--std", type=float, default=0.1, help="Standard deviation for initialization")
    parser.add_argument("--attack_result_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)