import matplotlib.pyplot as plt
import numpy as np

# Define k values
k_values = [1, 2, 3]

# Data for epsilon = 0.05
L_05 = [0.9864, 0.9952, 1.0027]
D_05 = [0.5315, 0.4958, 0.4484]
ASR_05 = [0.83, 0.73, 0.61]
Recall_05 = [0.16, 0.2, 0.2]
E2E_05 = [0.51, 0.565, 0.585]

# Data for epsilon = 0.08
L_08 = [0.9885, 0.9935, 1.0037]
D_08 = [0.5095, 0.4987, 0.5418]
ASR_08 = [0.83, 0.67, 0.53]
Recall_08 = [0.12, 0.215, 0.2067]
E2E_08 = [0.575, 0.635, 0.585]

# Plotting
fig, axs = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle('Comparison of Metrics at Different k and Epsilon Levels', fontsize=14)

metrics = ['L_top_k', 'D_top_k', 'Attack Success Rate', 'Recall@top-k', 'End-to-End@top-k']
data_05 = [L_05, D_05, ASR_05, Recall_05, E2E_05]
data_08 = [L_08, D_08, ASR_08, Recall_08, E2E_08]

for i, ax in enumerate(axs.flat[:5]):
    ax.plot(k_values, data_05[i], marker='o', label='epsilon=0.05')
    ax.plot(k_values, data_08[i], marker='s', label='epsilon=0.08')
    ax.set_title(metrics[i])
    ax.set_xlabel('k')
    ax.set_ylabel(metrics[i])
    ax.set_xticks(k_values)
    ax.grid(True)
    ax.legend()

# Hide the last subplot if unused
axs.flat[5].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()