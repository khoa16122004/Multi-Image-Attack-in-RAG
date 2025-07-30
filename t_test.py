import pandas as pd
from scipy.stats import ttest_ind_from_stats

# Đọc dữ liệu từ file CSV
df = pd.read_csv('final_results_std0.05.csv')

# Lặp qua từng model trong dataframe
for row in df.itertuples():
    print(f"Model: {row.model}")

    # Tìm mean cao nhất cho từng score (score_0 và score_1)
    score_0_means = [getattr(row, f'score_0_top{i}_mean') for i in range(1, 6)]
    score_1_means = [getattr(row, f'score_1_top{i}_mean') for i in range(1, 6)]

    score_0_max_mean = min(score_0_means)
    score_1_max_mean = min(score_1_means)

    # Tìm chỉ số top tương ứng với mean cao nhất
    score_0_max_index = score_0_means.index(score_0_max_mean) + 1
    score_1_max_index = score_1_means.index(score_1_max_mean) + 1

    # So sánh score_0_max_mean với từng top-k còn lại
    print(f"Comparing Score 0 (Max Mean) with other Top-k:")
    for i in range(1, 6):
        if i == score_0_max_index:
            continue  # Bỏ qua top có mean cao nhất
        score_0_mean = getattr(row, f'score_0_top{i}_mean')
        score_0_std = getattr(row, f'score_0_top{i}_std')
        score_0_n = getattr(row, f'score_0_top{i}_n', 30)

        t_stat, p_value = ttest_ind_from_stats(
            mean1=score_0_max_mean, std1=getattr(row, f'score_0_top{score_0_max_index}_std'), nobs1=getattr(row, f'score_0_top{score_0_max_index}_n', 30),
            mean2=score_0_mean, std2=score_0_std, nobs2=score_0_n
        )
        result = "PASS" if p_value < 0.05 else ""
        print(f"  Top-{i}: Mean = {score_0_mean:.3f}, T-statistic = {t_stat:.3f}, P-value = {p_value:.3e} {result}")

    # So sánh score_1_max_mean với từng top-k còn lại
    print(f"Comparing Score 1 (Max Mean) with other Top-k:")
    for i in range(1, 6):
        if i == score_1_max_index:
            continue  # Bỏ qua top có mean cao nhất
        score_1_mean = getattr(row, f'score_1_top{i}_mean')
        score_1_std = getattr(row, f'score_1_top{i}_std')
        score_1_n = getattr(row, f'score_1_top{i}_n', 30)

        t_stat, p_value = ttest_ind_from_stats(
            mean1=score_1_max_mean, std1=getattr(row, f'score_1_top{score_1_max_index}_std'), nobs1=getattr(row, f'score_1_top{score_1_max_index}_n', 30),
            mean2=score_1_mean, std2=score_1_std, nobs2=score_1_n
        )
        result = "PASS" if p_value < 0.05 else ""
        print(f"  Top-{i}: Mean = {score_1_mean:.3f}, T-statistic = {t_stat:.3f}, P-value = {p_value:.3e} {result}")

    print("-" * 60)