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
    [0.64, 0.69, 0.775, 0.74, 0.795],
    [0.625, 0.685, 0.73, 0.71, 0.715],
    [0.57, 0.635, 0.6383, 0.65, 0.675],
    [0.575, 0.615, 0.625, 0.645, 0.665],
    [0.57, 0.605, 0.625, 0.64, 0.66],
])

llava_next_adv = np.array([
    [0.71, 0.665, 0.69, 0.795, 0.73],
    [0.675, 0.65, 0.7, 0.705, 0.725],
    [0.6, 0.585, 0.645, 0.705, 0.77],
    [0.615, 0.565, 0.61, 0.65, 0.765],
    [0.625, 0.6, 0.6, 0.63, 0.745],
])

qwenvl2_adv = np.array([
    [0.925, 0.805, 0.775, 0.865, 0.87],
    [0.86, 0.865, 0.785, 0.78, 0.85],
    [0.83, 0.81, 0.87, 0.855, 0.85],
    [0.785, 0.795, 0.845, 0.865, 0.815],
    [0.775, 0.775, 0.835, 0.86, 0.85],
])

deepseek_vl2_adv = np.array([
    [0.75, 0.705, 0.75, 0.8, 0.77],
    [0.625, 0.725, 0.715, 0.69, 0.75],
    [0.61, 0.665, 0.685, 0.7, 0.715],
    [0.635, 0.615, 0.635, 0.7, 0.695],
    [0.62, 0.625, 0.63, 0.695, 0.665]
])

# Random data
llava_one_rand = np.array([
    [0.855, 0.87, 0.85, 0.86, 0.88],
    [0.88, 0.845, 0.845, 0.835, 0.845],
    [0.87, 0.825, 0.835, 0.82, 0.8],
    [0.86, 0.79, 0.8, 0.8, 0.855],
    [0.86, 0.79, 0.81, 0.82, 0.865],
])

llava_next_rand = np.array([
    [0.88, 0.8517, 0.82, 0.885, 0.89],
    [0.895, 0.83, 0.885, 0.825, 0.895],
    [0.94, 0.895, 0.89, 0.835, 0.895],
    [0.935, 0.885, 0.88, 0.845, 0.89],
    [0.915, 0.8, 0.87, 0.825, 0.92],
])

qwenvl2_rand = np.array([
    [0.91, 0.895, 0.915, 0.91, 0.895],
    [0.92, 0.895, 0.905, 0.885, 0.95],
    [0.92, 0.865, 0.91, 0.905, 0.915],
    [0.905, 0.865, 0.895, 0.895, 0.885],
    [0.89, 0.85, 0.895, 0.905, 0.88],
])

deepseek_vl2_rand = np.array([
    [0.88, 0.88, 0.885, 0.895, 0.92],
    [0.895, 0.845, 0.84, 0.83, 0.8],
    [0.855, 0.775, 0.83, 0.895, 0.875],
    [0.855, 0.775, 0.83, 0.895, 0.895],
    [0.855, 0.76, 0.8, 0.875, 0.875]
])

# Combine data for unified color scale
all_data = np.concatenate([
    llava_one_adv.flatten(), llava_next_adv.flatten(), 
    qwenvl2_adv.flatten(), deepseek_vl2_adv.flatten(),
    llava_one_rand.flatten(), llava_next_rand.flatten(), 
    qwenvl2_rand.flatten(), deepseek_vl2_rand.flatten()
])
vmin, vmax = all_data.min(), all_data.max()

models_data = {
    "LLaVA-One.": (llava_one_adv, llava_one_rand),
    "LLaVA-Next.": (llava_next_adv, llava_next_rand),
    "Qwen2.5VL": (qwenvl2_adv, qwenvl2_rand),
    "DeepseekVL2": (deepseek_vl2_adv, deepseek_vl2_rand),
}

# Create custom colormap - smaller values = darker colors
cmap = plt.cm.Reds_r

fig, axs = plt.subplots(2, 2, figsize=(16, 12))
axs = axs.flatten()

def add_small_rectangles(ax, data_rand, cmap, vmin, vmax):
    """Add small rectangles for random data in bottom-right corner of each cell"""
    for i in range(data_rand.shape[0]):
        for j in range(data_rand.shape[1]):
            # Normalize the random data value using unified scale
            norm_val = (data_rand[i, j] - vmin) / (vmax - vmin)
            color = cmap(norm_val)  # Reverse mapping so smaller values = darker
            
            rect = Rectangle((j + 0.6, i + 0.7), 0.4, 0.25, 
                            facecolor=color, edgecolor='white', linewidth=1)
            ax.add_patch(rect)

            # Vị trí chữ cũng điều chỉnh theo
            ax.text(j + 0.8, i + 0.825, f'{data_rand[i, j]:.3f}', 
                    ha='center', va='center', fontsize=12, color='black')
for ax, (name, (data_adv, data_rand)) in zip(axs, models_data.items()):
    # Create main heatmap with adversarial data
    sns.heatmap(
        data_adv,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        annot_kws={"size": 16},
        xticklabels=[1, 2, 3, 4, 5],
        yticklabels=[1, 2, 3, 4, 5],
        cbar=False,
        ax=ax,
        vmin=vmin,
        vmax=vmax
    )
    
    # Add small rectangles for random data
    add_small_rectangles(ax, data_rand, cmap, vmin, vmax)
    
    ax.set_title(name, fontsize=22)
    ax.set_xlabel("Retrieve Top-k images")
    ax.set_ylabel("Number of Injected Images")

# Add a single colorbar for the entire figure
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.tick_params(labelsize=14)

# Add legend
# from matplotlib.patches import Patch
# legend_elements = [
#     Patch(facecolor='lightgray', edgecolor='black', label='Adversarial (main)'),
#     Patch(facecolor='darkgray', edgecolor='white', label='Random (small rectangle)')
# ]
# fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
#            ncol=2, fontsize=16)

plt.tight_layout()
plt.subplots_adjust(bottom=0.08, right=0.92)

# Uncomment to save
plt.savefig("dual_data_heatmap.pdf", dpi=300, bbox_inches='tight')
# plt.show()