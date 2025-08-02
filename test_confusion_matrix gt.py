import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rc('legend', fontsize=20)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=20)

# Adversarial data
llava_one_adv = np.array([
    [0.28, 0.33, 0.31, 0.28, 0.305],
    [0.285, 0.355, 0.325, 0.285, 0.33],
    [0.285, 0.355, 0.325, 0.305, 0.315],
    [0.285, 0.34, 0.305, 0.305, 0.33],
    [0.265, 0.33, 0.315, 0.28, 0.315],
])

llava_next_adv = np.array([
    [0.275, 0.32, 0.315, 0.31, 0.265],
    [0.3, 0.34, 0.32, 0.32, 0.26],
    [0.3, 0.325, 0.345, 0.32, 0.25],
    [0.29, 0.33, 0.335, 0.305, 0.295],
    [0.215, 0.345, 0.33, 0.325, 0.285],
])

qwenvl2_adv = np.array([
    [0.355, 0.335, 0.375, 0.385, 0.375],
    [0.28, 0.335, 0.34, 0.4, 0.36],
    [0.28, 0.345, 0.365, 0.405, 0.365],
    [0.315, 0.34, 0.36, 0.395, 0.375],
    [0.29, 0.35, 0.37, 0.39, 0.4]
])

deepseek_vl2_adv = np.array([
    [0.305, 0.3, 0.29, 0.275, 0.31],
    [0.3, 0.32, 0.295, 0.285, 0.27],
    [0.295, 0.315, 0.29, 0.295, 0.3],
    [0.28, 0.32, 0.3, 0.295, 0.29],
    [0.29, 0.295, 0.285, 0.28, 0.28]  
])

# Loại bỏ các model có None
models_data = {
    "LLaVA-One.": llava_one_adv,
    "LLaVA-Next.": llava_next_adv,
    "Qwen2.5VL": qwenvl2_adv,
    "DeepseekVL2": deepseek_vl2_adv,
}
# Lọc các model có dữ liệu đầy đủ (không có None trong bất kỳ dòng nào)
valid_models_data = {}
for name, data in models_data.items():
    if isinstance(data, np.ndarray):
        valid_models_data[name] = data
    elif isinstance(data, list):
        if all(isinstance(row, np.ndarray) for row in data):
            valid_models_data[name] = np.stack(data)

# Tính vmin, vmax
all_values = []
for data in valid_models_data.values():
    all_values.extend(data.flatten())
vmin, vmax = np.min(all_values), np.max(all_values)




# Plot
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs = axs.flatten()

cmap = plt.cm.bwr_r
for ax, (name, data_adv) in zip(axs, valid_models_data.items()):
    labels = np.vectorize(lambda x: f'{x*100:.1f}')(data_adv)

    sns.heatmap(
        data_adv,
        annot=labels,
        fmt="",
        cmap=cmap,
        annot_kws={"size": 18},
        xticklabels=[1, 2, 3, 4, 5],
        yticklabels=[1, 2, 3, 4, 5],
        cbar=False,
        ax=ax,
        vmin=vmin,
        vmax=vmax
    )
    ax.set_title(name, fontsize=22)
    ax.set_xlabel("Retrieve Top-k images")
    ax.set_ylabel("Number of Injected Images")

# Colorbar
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.tick_params(labelsize=14)

plt.tight_layout()
plt.subplots_adjust(bottom=0.08, right=0.92)
plt.savefig("adv_only_heatmap.pdf", dpi=300, bbox_inches='tight')
# plt.show()

