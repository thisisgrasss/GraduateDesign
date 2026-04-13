import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

# 数据集根目录
dataset_root = r"/root/autodl-tmp/Grounded_sam/Grounded-Segment-Anything/dataset"

# 支持的图像格式
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

def get_images(folder, n=5):
    """获取文件夹内前 n 张图像路径"""
    files = sorted([
        f for f in Path(folder).iterdir()
        if f.suffix.lower() in IMG_EXTS
    ])
    return files[:n]

# 遍历所有子文件夹（保持层级顺序）
folders = []
for root, dirs, files in os.walk(dataset_root):
    dirs.sort()  # 保证遍历顺序一致
    # 只收集叶子节点（含图像的文件夹）
    images = [f for f in Path(root).iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS]
    if images:
        folders.append(Path(root))

n_cols = 5
n_rows = len(folders)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(n_cols * 3, n_rows * 3.2)
)

# 确保 axes 始终是二维数组
if n_rows == 1:
    axes = [axes]

for row_idx, folder in enumerate(folders):
    images = get_images(folder, n=n_cols)
    
    # 行标签：取相对路径作为标题
    rel = folder.relative_to(dataset_root)
    row_label = str(rel).replace("\\", " / ")

    for col_idx in range(n_cols):
        ax = axes[row_idx][col_idx]
        if col_idx < len(images):
            img = mpimg.imread(str(images[col_idx]))
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(images[col_idx].name, fontsize=7, pad=3)
        else:
            ax.axis("off")  # 图像不足时留空
        ax.set_xticks([])
        ax.set_yticks([])

    # 在每行最左侧添加文件夹名称
    axes[row_idx][0].set_ylabel(row_label, fontsize=9, fontweight="bold",
                                 labelpad=8, rotation=0,
                                 ha="right", va="center")

plt.suptitle("Dataset Preview — First 5 Images per Folder",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("dataset_preview.png", dpi=150, bbox_inches="tight")
plt.show()
print("预览图已保存至 dataset_preview.png")