import matplotlib.pyplot as plt

# Dữ liệu
topks = [1, 2, 3, 4, 5]
scores_1 = [0.555, 0.69, 0.7, 0.72, 0.785]
scores_2 = [0.6, 0.63, 0.69, 0.6983, 0.705]
scores_3 = [0.55, 0.635, 0.655, 0.62, 0.655]

# Vẽ biểu đồ
plt.figure(figsize=(8, 5))
plt.plot(topks, scores_1, marker='o', label='Inject top-1 document')
plt.plot(topks, scores_2, marker='o', label='Inject top-2 documents')
plt.plot(topks, scores_3, marker='o', label='Inject top-3 documents')

# Cấu hình biểu đồ
plt.xlabel("Top-k")
plt.ylabel("End-to-End Accuracy")
plt.title("So sánh End-to-End Accuracy theo số lượng document được inject")
plt.xticks(topks)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
