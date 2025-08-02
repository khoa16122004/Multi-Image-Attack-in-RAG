import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle

llava_one_adv = np.array([
    [0.7857, 0.7571,  0.7821 ,0.8000, 0.8333],
    [0.4857142857142857, 0.5285714285714286, 0.5428571428571428, 0.5, 0.5285714285714286],
    [0.4857142857142857, 0.5857142857142857, 0.5, 0.4857142857142857, 0.5571428571428572],
    [0.4358974358974359, 0.5897435897435898, 0.5512820512820513, 0.5, 0.5641025641025641],
    [0.45714285714285713, 0.5714285714285714, 0.5428571428571428, 0.5714285714285714, 0.6285714285714286],
    [0.4128205128205128, 0.6025641025641025, 0.6410256410256411, 0.5512820512820513, 0.6153846153846154]
])


llava_next_adv = np.array([
    [ 0.7778, 0.8281, 0.8226, 0.8269, 0.8229],
    [0.5555555555555556, 0.5, 0.5370370370370371, 0.5555555555555556, 0.4444444444444444],
    [0.46875, 0.525, 0.546875, 0.546875, 0.4375],
    [0.43548387096774194, 0.5645161290322581, 0.6290322580645161, 0.5806451612903226, 0.46774193548387094],
    [0.4230769230769231, 0.5576923076923077, 0.5192307692307693, 0.5769230769230769, 0.5769230769230769],
    [0.39655172413793105, 0.5, 0.5517241379310345, 0.5689655172413793, 0.5344827586206896]
])

qwenvl2_adv = np.array([
    [0.8500, 0.7935, 0.8696, 0.8152,  0.8229] ,
    [0.65, 0.6125, 0.6625, 0.6375, 0.6125],
    [0.6875, 0.6, 0.625, 0.65, 0.6],
    [0.6625, 0.6375, 0.6625, 0.675, 0.6125],
    [0.65, 0.6125, 0.6625, 0.6625, 0.6375],
    [0.675, 0.6125, 0.6375, 0.65, 0.6625]
])

deepseek_vl2_adv = np.array([
    [0.7969, 0.7778, 0.8500, 0.7969, 0.8065], 
    [0.65, 0.6666666666666666, 0.6166666666666667, 0.6666666666666666, 0.6333333333333333],
    [0.6333333333333333, 0.6333333333333333, 0.5833333333333334, 0.6, 0.6333333333333333],
    [0.6333333333333333, 0.6333333333333333, 0.55, 0.6166666666666667, 0.65],
    [0.6333333333333333, 0.5833333333333334, 0.55, 0.6166666666666667, 0.6333333333333333],
    [0.65, 0.6, 0.55, 0.6, 0.6166666666666667]

])


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
        yticklabels=[0, 1, 2, 3, 4, 5],
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
plt.savefig("gt_with_indvidual_set.pdf", dpi=300, bbox_inches='tight')