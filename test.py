import matplotlib.pyplot as plt

# Dữ liệu
clean_rates = {
    1: [0.13, 0.515, 0.6733, 0.755, 0.8],
    2: [0.05, 0.205, 0.38, 0.5175, 0.61],
    3: [0.04, 0.095, 0.21, 0.32, 0.43]
}

mrr_scores = {
    1: [0.13, 0.5575, 0.4489, 0.3832, 0.3390],
    2: [0.05, 0.23, 0.41, 0.3386, 0.2948],
    3: [0.04, 0.115, 0.2508, 0.3275, 0.2755]
}

top_k = [1, 2, 3, 4, 5]
colors = ['blue', 'green', 'red']

# Vẽ biểu đồ Clean Rate
plt.figure(figsize=(10, 4))
for i, k in enumerate(clean_rates):
    plt.plot(top_k, clean_rates[k], label=f'k={k}', marker='o', color=colors[i])
plt.title("Clean Rate per Top-k Position")
plt.xlabel("Top-k Position")
plt.ylabel("Clean Rate")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Vẽ biểu đồ MRR Score
plt.figure(figsize=(10, 4))
for i, k in enumerate(mrr_scores):
    plt.plot(top_k, mrr_scores[k], label=f'k={k}', marker='o', color=colors[i])
plt.title("MRR Score per Top-k Position")
plt.xlabel("Top-k Position")
plt.ylabel("MRR Score")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
