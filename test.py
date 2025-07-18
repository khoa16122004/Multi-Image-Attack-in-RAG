import matplotlib.pyplot as plt

positions = [1, 2, 3, 4, 5]

clean_rate = {
    1: [0.13, 0.515, 0.6733333333333326, 0.755, 0.7999999999999985],
    2: [0.05, 0.205, 0.37999999999999984, 0.5175, 0.6100000000000009],
    3: [0.04, 0.095, 0.20999999999999988, 0.32, 0.4299999999999994],
    0: [1] * 5,
}

mrr_score = {
    1: [0.13, 0.5575, 0.44888888888888867, 0.3831944444444447, 0.33899999999999963],
    2: [0.05, 0.23, 0.4100000000000001, 0.3386111111111114, 0.2948472222222225],
    3: [0.04, 0.115, 0.2508333333333332, 0.3275, 0.2754722222222224],
    0: [1] * 5,
}

colors = {
    0: "black",
    1: "blue",
    2: "green",
    3: "red"
}

labels = {
    0: "k=0 (not poision)",
    1: "Inject k=1",
    2: "Inject k=2",
    3: "Inject k=3"
}

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot Clean Rate
for k in [0, 1, 2, 3]:
    axs[0].plot(positions, clean_rate[k], label=labels[k], color=colors[k], marker="o")
axs[0].set_title("Mean Clean Rate per Top-k Position")
axs[0].set_xlabel("Top-k Position")
axs[0].set_ylabel("Clean Rate")
axs[0].set_xticks(positions)
axs[0].legend()
axs[0].grid(True)

# Plot MRR Score
for k in [0, 1, 2, 3]:
    axs[1].plot(positions, mrr_score[k], label=labels[k], color=colors[k], marker="o")
axs[1].set_title("Mean MRR Score per Top-k Position")
axs[1].set_xlabel("Top-k Position")
axs[1].set_ylabel("MRR Score")
axs[1].set_xticks(positions)
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
