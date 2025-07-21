import matplotlib.pyplot as plt

topk = [1, 2, 3, 4, 5]

# Trung bình các mô hình
avg_clean_rate = [0.0425, 0.1075, 0.218333, 0.321875, 0.4375]
avg_mrr = [0.0425, 0.12875, 0.259291, 0.332465, 0.278691]

# Không attack (baseline)
ideal_clean_rate = [1, 2, 3, 4, 5]   # Nếu tính số ảnh sạch tìm được
ideal_mrr = [1, 1+1/2, 1+1/2+1/3, 1+1/2+1/3+1/4, 1+1/2+1/3+1/4+1/5]

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(topk, avg_clean_rate, marker='o', label='Injected (average)', color='red')
plt.plot(topk, ideal_clean_rate, marker='o', label='No attack (ideal)', color='green')
plt.title("Clean Rate vs Top‑k")
plt.xlabel("Top‑k")
plt.ylabel("Clean Rate")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(topk, avg_mrr, marker='o', label='Injected (average)', color='red')
plt.plot(topk, ideal_mrr, marker='o', label='No attack (ideal)', color='green')
plt.title("MRR vs Top‑k")
plt.xlabel("Top‑k")
plt.ylabel("MRR Score")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
