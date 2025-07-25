import matplotlib.pyplot as plt
import numpy as np

# Thiết lập font
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rc('legend', fontsize=25)  # Giảm từ 25 xuống 20
plt.rc('xtick', labelsize=25)  # Giảm từ 25 xuống 20
plt.rc('ytick', labelsize=25)  # Giảm từ 25 xuống 20
plt.rc('axes', labelsize=24)   # Giảm từ 30 xuống 24

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

deepseek_vl2 = [
    [0.75, 0.705, 0.75, 0.8, 0.77],
    [0.625, 0.725, 0.715, 0.69, 0.75],
    [0.61, 0.665, 0.685, 0.7, 0.715],
    [0.635, 0.615, 0.635, 0.7, 0.695],
    [0.62, 0.625, 0.63, 0.695, 0.665]
]

mean_llava_one = np.mean(llava_one, axis=1)
mean_llava_next = np.mean(llava_next, axis=1)
mean_qwenvl2 = np.mean(qwenvl2, axis=1)
mean_deepseekvl2 = np.mean(deepseek_vl2, axis=1)

injects = [1, 2, 3, 4, 5]

# Vẽ biểu đồ với chiều ngang thu hẹp
plt.figure(figsize=(7, 6))

# Vẽ các đường
plt.plot(injects, mean_llava_one, marker='o', linestyle='-', linewidth=3, markersize=10, color='#b30000', label='LLaVA-One.')
plt.plot(injects, mean_llava_next, marker='s', linestyle='--', linewidth=3, markersize=10, color='#004c6d', label='LLaVA-Next.')
plt.plot(injects, mean_qwenvl2, marker='^', linestyle='-.', linewidth=3, markersize=10, color='#3f007d', label='Qwen2.5VL')
plt.plot(injects, mean_deepseekvl2, marker='D', linestyle=':', linewidth=3, markersize=10, color='#8c564b', label='DeepseekVL2')

# Thêm text với vị trí được sắp xếp hợp lý
for i, x in enumerate(injects):
    # Lấy giá trị của tất cả các model tại điểm x
    values = [
        (mean_qwenvl2[i], 'Qwen2.5VL', '#3f007d'),
        (mean_deepseekvl2[i], 'DeepseekVL2', '#8c564b'),
        (mean_llava_one[i], 'LLaVA-One.', '#b30000'),
        (mean_llava_next[i], 'LLaVA-Next.', '#004c6d')
    ]
    
    # Đặt text cho từng model với vị trí tùy chỉnh
    for value, model_name, color in values:
        if model_name == 'Qwen2.5VL':
            # Qwen luôn ở phía trên
            plt.text(
                x, value + 0.015, f'{value:.4f}',
                ha='center', va='bottom',
                fontsize=10, color=color, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7, boxstyle='round,pad=0.3')
            )
        elif model_name == 'DeepseekVL2':
            # Deepseek ở phía trên (thấp hơn Qwen một chút)
            plt.text(
                x, value + 0.008, f'{value:.4f}',
                ha='center', va='bottom',
                fontsize=10, color=color, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7, boxstyle='round,pad=0.3')
            )
        elif model_name == 'LLaVA-One.' and x == 1:
            # LLaVA-One ở x=1 nằm bên trái
            plt.text(
                x - 0.08, value, f'{value:.4f}',
                ha='right', va='center',
                fontsize=10, color=color, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7, boxstyle='round,pad=0.3')
            )
        elif model_name == 'LLaVA-Next.' and x == 3:
            # LLaVA-Next ở x=3 nằm bên trái
            plt.text(
                x - 0.08, value, f'{value:.4f}',
                ha='right', va='center',
                fontsize=10, color=color, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7, boxstyle='round,pad=0.3')
            )
        elif model_name == 'LLaVA-One.' and x == 2:
            # LLaVA-One ở x=2 nằm bên trái
            plt.text(
                x - 0.08, value, f'{value:.4f}',
                ha='right', va='center',
                fontsize=10, color=color, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7, boxstyle='round,pad=0.3')
            )
        elif model_name == 'LLaVA-Next.' and x == 2:
            # LLaVA-Next ở x=2 nằm phía dưới
            plt.text(
                x, value - 0.015, f'{value:.4f}',
                ha='center', va='top',
                fontsize=10, color=color, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7, boxstyle='round,pad=0.3')
            )
        elif model_name == 'LLaVA-Next.' and x in [4, 5]:
            # LLaVA-Next ở x=4,5 nằm bên trái
            plt.text(
                x - 0.08, value, f'{value:.4f}',
                ha='right', va='center',
                fontsize=10, color=color, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7, boxstyle='round,pad=0.3')
            )
        elif model_name == 'LLaVA-One.' and x in [4, 5]:
            # LLaVA-One ở x=4,5 nằm phía dưới
            plt.text(
                x, value - 0.015, f'{value:.4f}',
                ha='center', va='top',
                fontsize=10, color=color, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7, boxstyle='round,pad=0.3')
            )
        else:
            # Các trường hợp còn lại đặt phía dưới
            plt.text(
                x, value - 0.015, f'{value:.4f}',
                ha='center', va='top',
                fontsize=10, color=color, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7, boxstyle='round,pad=0.3')
            )

# Trang trí
plt.xlabel('Number of Adversarial Images', fontsize=16)
plt.ylabel('Mean End-to-End Score across Top-k', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(injects)
plt.ylim(0.55, 0.95)

# Legend với font size phù hợp
plt.legend(frameon=True, loc='upper right', fontsize=14)
plt.tight_layout()

# plt.show()
save_path = f"visualization/std{0.05}_number_injection.pdf"
plt.savefig(save_path, dpi=300, bbox_inches='tight')