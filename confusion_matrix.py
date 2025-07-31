import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rc('legend', fontsize=20)
plt.rc('xtick', labelsize=14)  # Tăng từ 12 lên 14
plt.rc('ytick', labelsize=14)  # Tăng từ 12 lên 14
plt.rc('axes', labelsize=20)   # Tăng từ 18 lên 20

# adversarial 
llava_one = np.array([
    [0.64, 0.69, 0.775, 0.74, 0.795],
    [0.625, 0.685, 0.73, 0.71, 0.715],
    [0.57, 0.635, 0.6383, 0.65, 0.675],
    [0.575, 0.615, 0.625, 0.645, 0.665],
    [0.57, 0.605, 0.625, 0.64, 0.66],
])

llava_next = np.array([
    [0.71, 0.665, 0.69, 0.795, 0.73],
    [0.675, 0.65, 0.7, 0.705, 0.725],
    [0.6, 0.585, 0.645, 0.705, 0.77],
    [0.615, 0.565, 0.61, 0.65, 0.765],
    [0.625, 0.6, 0.6, 0.63, 0.745],
])

qwenvl2 = np.array([
    [0.925, 0.805, 0.775, 0.865, 0.87],
    [0.86, 0.865, 0.785, 0.78, 0.85],
    [0.83, 0.81, 0.87, 0.855, 0.85],
    [0.785, 0.795, 0.845, 0.865, 0.815],
    [0.775, 0.775, 0.835, 0.86, 0.85],
])

deepseek_vl2 = np.array([
    [0.75, 0.705, 0.75, 0.8, 0.77],
    [0.625, 0.725, 0.715, 0.69, 0.75],
    [0.61, 0.665, 0.685, 0.7, 0.715],
    [0.635, 0.615, 0.635, 0.7, 0.695],
    [0.62, 0.625, 0.63, 0.695, 0.665]
])

# # random
llava_one = np.array([
    [0.855, 0.87, 0.85, 0.86, 0.88],
    [0.88, 0.845, 0.845, 0.835, 0.845],
    [0.87, 0.825, 0.835, 0.82, 0.8],
    [0.86, 0.79, 0.8, 0.8, 0.855],
    [0.86, 0.79, 0.81, 0.82, 0.865],
])

llava_next = np.array([
    [0.88, 0.8517, 0.82, 0.885, 0.89],
    [0.895, 0.83, 0.885, 0.825, 0.895],
    [0.94, 0.895, 0.89, 0.835, 0.895],
    [0.935, 0.885, 0.88, 0.845, 0.89],
    [0.915, 0.8, 0.87, 0.825, 0.92],
])

qwenvl2 = np.array([
    [0.91, 0.895, 0.915, 0.91, 0.895],
    [0.92, 0.895, 0.905, 0.885, 0.95],
    [0.92, 0.865, 0.91, 0.905, 0.915],
    [0.905, 0.865, 0.895, 0.895, 0.885],
    [0.89, 0.85, 0.895, 0.905, 0.88],
])

deepseek_vl2 = np.array([
    [0.88, 0.88, 0.885, 0.895, 0.92],
    [0.895, 0.845, 0.84, 0.83, 0.8],
    [0.855, 0.775, 0.83, 0.895, 0.875],
    [0.855, 0.775, 0.83, 0.895, 0.895],
    [0.855, 0.76, 0.8, 0.875, 0.875]
])





models = {
    "LLaVA-One.": llava_one,
    "LLaVA-Next.": llava_next,
    "Qwen2.5VL": qwenvl2,
    "DeepseekVL2": deepseek_vl2,
}

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for ax, (name, data) in zip(axs, models.items()):
    sns.heatmap(
        data,
        annot=data,
        fmt=".3f",
        cmap="Reds_r",                   # Đổi sang đỏ
        annot_kws={"size": 20},        # Tăng size chữ trong ô
        xticklabels=[1, 2, 3, 4, 5],
        yticklabels=[1, 2, 3, 4, 5],
        cbar=False,
        ax=ax
    )
    ax.set_title(name, fontsize=22)
    ax.set_xlabel("Retrieve Top-k images")
    ax.set_ylabel("Number of Injected Images")

plt.tight_layout()
# plt.savefig("confusion_matrix_red.pdf", dpi=300)
plt.show()