import matplotlib.pyplot as plt
import numpy as np

# Thiết lập font
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=20)

# Dữ liệu
llava_one = [
    [0.64, 0.69, 0.775, 0.74, 0.795],
    [0.625, 0.685, 0.73, 0.71, 0.715],
    [0.57, 0.635, 0.6383, 0.65, 0.675],
    [0.575, 0.615, 0.625, 0.645, 0.665],
    [0.57, 0.605, 0.625, 0.64, 0.66],
]

llava_next = [
    [0.71, 0.665, 0.69, 0.795, 0.73],
    [0.675, 0.65, 0.7, 0.705, 0.725],
    [0.6, 0.585, 0.645, 0.705, 0.77],
    [0.615, 0.565, 0.61, 0.65, 0.765],
    [0.625, 0.6, 0.6, 0.63, 0.745],
]

qwenvl2 = [
    [0.925, 0.805, 0.775, 0.865, 0.87],
    [0.86, 0.865, 0.785, 0.78, 0.85],
    [0.83, 0.81, 0.87, 0.855, 0.85],
    [0.785, 0.795, 0.845, 0.865, 0.815],
    [0.775, 0.775, 0.835, 0.86, 0.85],
]

mean_llava_one = np.mean(llava_one, axis=1)
mean_llava_next = np.mean(llava_next, axis=1)
mean_qwenvl2 = np.mean(qwenvl2, axis=1)

injects = [1, 2, 3, 4, 5]

# Hàm vẽ đường + text theo màu của line
def plot_with_values(x, y, label, color, marker, linestyle, text_offset=0.015, decimal=4):
    plt.plot(x, y, marker=marker, linestyle=linestyle, linewidth=2, markersize=8, color=color, label=label)
    for xi, yi in zip(x, y):
        va = 'bottom' if text_offset > 0 else 'top'
        plt.text(
            xi, yi + text_offset, f'{yi:.{decimal}f}',
            ha='center', va=va,
            fontsize=9, color=color,
            fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.6, boxstyle='round,pad=0.25')
        )

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))

plot_with_values(injects, mean_llava_one, 'LLaVA-One', '#b30000', 'o', '-', text_offset=0.018)
plot_with_values(injects, mean_llava_next, 'LLaVA-Next', '#004c6d', 's', '--', text_offset=0.035)
plot_with_values(injects, mean_qwenvl2, 'Qwen-VL-2', '#3f007d', '^', '-.', text_offset=-0.035)

# Trang trí
plt.xlabel('Number of Positioning Images', fontsize=14)
plt.ylabel('Mean End-to-End Score across Top-k', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(injects)
plt.ylim(0.55, 0.95)

# Legend ở góc trái dưới trong biểu đồ
plt.legend(frameon=True, loc='lower left')
plt.tight_layout()

plt.show()
