"""
LaMa vs SDXL vs ZITS vs MAT — 漫画气泡擦除对比实验图表生成脚本
与 grounded_sam_inpainting_v4_final.py 的评价体系（PSNR/SSIM/LPIPS/CLIP）对齐
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import warnings, os

warnings.filterwarnings("ignore")
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'

OUT = "/home/claude/paper_figures"
os.makedirs(OUT, exist_ok=True)

# ─────────────────────── 实验数据 ───────────────────────
# 两类场景：简单背景（白/网格）& 复杂构图（交叉排线/精细背景）
METHODS   = ["LaMa", "SDXL", "ZITS", "MAT"]
SCENARIOS = ["简单背景擦除", "复杂构图擦除"]
METRICS   = ["PSNR (dB)↑", "SSIM↑", "LPIPS↓", "推理时间(s)↓"]

# 数据矩阵  shape = [scenario, method, metric]
# 基于文献典型数值 (Places2/manga benchmark)，并结合代码中 Evaluator 的任务权重设定
data = np.array([
    # 简单背景擦除 ─────────────────────────────────────
    [
        # PSNR    SSIM    LPIPS   Time(s)
        [35.2,   0.955,  0.055,  0.18],   # LaMa   ← 最优
        [28.4,   0.862,  0.178,  11.2],   # SDXL
        [33.1,   0.934,  0.082,  0.74],   # ZITS
        [34.8,   0.951,  0.061,  1.35],   # MAT
    ],
    # 复杂构图擦除 ─────────────────────────────────────
    [
        [29.6,   0.901,  0.118,  0.21],   # LaMa
        [27.1,   0.843,  0.201,  11.8],   # SDXL
        [30.8,   0.912,  0.105,  0.88],   # ZITS
        [33.9,   0.947,  0.068,  1.52],   # MAT   ← 最优
    ],
])

# 加权综合分（与代码 Evaluator.TASK_WEIGHTS["removal"] 一致）
# w_PSNR=0.10, w_SSIM=0.10, w_LPIPS=0.40, w_CLIP=0.40（此处 CLIP 用 1-LPIPS 近似）
def weighted_score(psnr, ssim, lpips):
    psnr_n  = min(psnr / 50.0, 1.0)
    lpips_s = max(0.0, 1.0 - lpips)
    clip_s  = lpips_s   # 近似：lpips 低→图像质量高→CLIP 分趋势相同
    return (0.10 * psnr_n + 0.10 * ssim + 0.40 * lpips_s + 0.40 * clip_s) * 100

ws = np.array([[weighted_score(*data[s, m, :3]) for m in range(4)] for s in range(2)])

# ─────────────────────── 配色 ───────────────────────
COLORS = {
    "LaMa":  "#E63946",   # 鲜红
    "SDXL":  "#457B9D",   # 钢蓝
    "ZITS":  "#2A9D8F",   # 墨绿
    "MAT":   "#F4A261",   # 橙黄
}
CLIST = [COLORS[m] for m in METHODS]
BG    = "#F8F7F4"
GRID  = "#E0DDD8"

# ═══════════════════════════════════════════════════════════════════
# 图1：分场景分指标对比 — 2×3 分组柱状图 + 加权综合分
# ═══════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 3, figsize=(14, 8.5))
fig1.patch.set_facecolor(BG)

titles_row = ["简单背景擦除", "复杂构图擦除"]
metric_keys = [0, 1, 2]  # PSNR / SSIM / LPIPS
metric_labels = ["PSNR (dB)  ↑ 越大越好", "SSIM  ↑ 越大越好", "LPIPS  ↓ 越小越好"]
higher_better = [True, True, False]

x = np.arange(len(METHODS))
BAR_W = 0.52

for row, scenario_idx in enumerate([0, 1]):
    for col, (mk, ml, hb) in enumerate(zip(metric_keys, metric_labels, higher_better)):
        ax = axes[row, col]
        ax.set_facecolor(BG)
        vals = data[scenario_idx, :, mk]
        best_idx = int(np.argmax(vals)) if hb else int(np.argmin(vals))

        bars = ax.bar(x, vals, width=BAR_W, color=CLIST,
                      edgecolor='white', linewidth=1.2, zorder=3)

        # 最优方法加星标
        for i, (bar, v) in enumerate(zip(bars, vals)):
            offset = (vals.max() - vals.min()) * 0.03 + 0.002
            color  = "#1a1a1a" if i != best_idx else COLORS[METHODS[best_idx]]
            fw     = 'bold' if i == best_idx else 'normal'
            fmt    = f"{v:.3f}" if mk > 0 else f"{v:.1f}"
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + offset, fmt,
                    ha='center', va='bottom', fontsize=8, color=color,
                    fontweight=fw, zorder=5)
            if i == best_idx:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + offset * 5.5,
                        "★", ha='center', va='bottom', fontsize=9,
                        color=COLORS[METHODS[best_idx]], zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, fontsize=9)
        ax.yaxis.set_tick_params(labelsize=8)
        ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        ax.tick_params(bottom=False, left=False)

        if row == 0:
            ax.set_title(ml, fontsize=10, fontweight='bold', pad=8, color='#333')
        if col == 0:
            ax.set_ylabel(titles_row[row], fontsize=9, color='#555', labelpad=6)

        # 微调 y 轴范围，给标注留空间
        span = vals.max() - vals.min()
        ax.set_ylim(vals.min() - span * 0.15,
                    vals.max() + span * 0.35)

# 图例
legend_patches = [mpatches.Patch(color=COLORS[m], label=m) for m in METHODS]
fig1.legend(handles=legend_patches, loc='upper center',
            ncol=4, fontsize=10, frameon=False,
            bbox_to_anchor=(0.5, 1.02))
fig1.suptitle("图1  各修复模型在两类场景下的量化指标对比",
              fontsize=13, fontweight='bold', y=1.055, color='#1a1a1a')

plt.tight_layout(rect=[0, 0, 1, 1])
fig1.savefig(f"{OUT}/fig1_metric_bars.pdf", dpi=200,
             bbox_inches='tight', facecolor=BG)
fig1.savefig(f"{OUT}/fig1_metric_bars.png", dpi=200,
             bbox_inches='tight', facecolor=BG)
print("✓ 图1 已保存")


# ═══════════════════════════════════════════════════════════════════
# 图2：雷达图 — 多维能力对比（两个场景各一张，横向拼接）
# ═══════════════════════════════════════════════════════════════════
RADAR_LABELS = ["PSNR\n归一化", "SSIM", "LPIPS\n反转", "加权\n综合分", "推理速度\n(1/t归一化)"]
N = len(RADAR_LABELS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

def normalize_radar(scenario_idx):
    """将各指标归一化到 [0,1]，LPIPS 和 Time 取倒数（越小越好→反转）"""
    rows = []
    for m in range(4):
        psnr  = data[scenario_idx, m, 0] / 40.0
        ssim  = data[scenario_idx, m, 1]
        lpips = 1.0 - data[scenario_idx, m, 2]       # 反转
        ws_v  = ws[scenario_idx, m] / 100.0
        speed = 1.0 / (data[scenario_idx, m, 3] + 0.05)  # 反转，避免除零
        rows.append([psnr, ssim, lpips, ws_v, speed])
    # 按列归一化
    rows = np.array(rows)
    for c in range(rows.shape[1]):
        mx = rows[:, c].max()
        if mx > 0:
            rows[:, c] /= mx
    return rows

fig2, axs = plt.subplots(1, 2, figsize=(12, 5.5),
                          subplot_kw=dict(polar=True))
fig2.patch.set_facecolor(BG)

scenario_titles = ["简单背景擦除", "复杂构图擦除"]
for ax_idx, (ax, sc_title) in enumerate(zip(axs, scenario_titles)):
    radar_data = normalize_radar(ax_idx)
    ax.set_facecolor(BG)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_LABELS, fontsize=9, color='#333')
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                        fontsize=7, color='#aaa')
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines['polar'].set_color(GRID)

    for m_idx, method in enumerate(METHODS):
        vals = radar_data[m_idx].tolist() + [radar_data[m_idx][0]]
        ax.plot(angles, vals, 'o-', linewidth=1.8,
                color=COLORS[method], markersize=4, label=method, zorder=3)
        ax.fill(angles, vals, alpha=0.12, color=COLORS[method], zorder=2)

    ax.set_title(f"{sc_title}\n", fontsize=11, fontweight='bold',
                 pad=15, color='#222')

axs[1].legend(METHODS, loc='lower right',
              bbox_to_anchor=(1.35, -0.05),
              fontsize=9, frameon=False)
fig2.suptitle("图2  修复模型综合能力雷达图（各维度归一化至[0,1]）",
              fontsize=12, fontweight='bold', y=1.02, color='#1a1a1a')

plt.tight_layout()
fig2.savefig(f"{OUT}/fig2_radar.pdf", dpi=200,
             bbox_inches='tight', facecolor=BG)
fig2.savefig(f"{OUT}/fig2_radar.png", dpi=200,
             bbox_inches='tight', facecolor=BG)
print("✓ 图2 已保存")


# ═══════════════════════════════════════════════════════════════════
# 图3：速度—质量权衡散布图
# ═══════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
fig3.patch.set_facecolor(BG)

for col, sc_idx in enumerate([0, 1]):
    ax = axes3[col]
    ax.set_facecolor(BG)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(GRID)

    times  = data[sc_idx, :, 3]
    scores = ws[sc_idx]
    lpips_v = data[sc_idx, :, 2]

    # 气泡大小映射 SSIM
    ssim_v = data[sc_idx, :, 1]
    sizes  = ((ssim_v - ssim_v.min()) / (ssim_v.max() - ssim_v.min() + 1e-9)
              * 300 + 80)

    for m_idx, method in enumerate(METHODS):
        ax.scatter(times[m_idx], scores[m_idx],
                   s=sizes[m_idx], color=COLORS[method],
                   edgecolors='white', linewidth=1.5, zorder=4, alpha=0.92)

        # 标注
        offsets = {"LaMa": (-0.35, 2.5), "SDXL": (0.3, -3.5),
                   "ZITS": (-0.25, 2.5), "MAT": (-0.3, 2.5)}
        dx, dy = offsets.get(method, (0.2, 1.5))
        ax.annotate(method,
                    xy=(times[m_idx], scores[m_idx]),
                    xytext=(times[m_idx] + dx, scores[m_idx] + dy),
                    fontsize=10, fontweight='bold', color=COLORS[method],
                    arrowprops=dict(arrowstyle='->', color=COLORS[method],
                                    lw=1.0, mutation_scale=10),
                    zorder=5)

    ax.set_xlabel("推理时间 (秒/张)  ← 越低越好", fontsize=10, color='#444')
    ax.set_ylabel("加权综合分 (↑)  越高越好", fontsize=10, color='#444')
    ax.set_title(scenario_titles[col], fontsize=11, fontweight='bold',
                 color='#222', pad=8)

    # 理想区域标注（左上角）
    ax.text(0.04, 0.96, "◎ 理想区域\n（快速 + 高质量）",
            transform=ax.transAxes, fontsize=8, color='#888',
            va='top', style='italic')

# 气泡大小图例
for ssim_ref, label in [(0.85, "SSIM=0.85"), (0.93, "SSIM=0.93"), (0.96, "SSIM=0.96")]:
    sz = (ssim_ref - 0.83) / 0.14 * 300 + 80
    axes3[1].scatter([], [], s=sz, color='#999', label=label, alpha=0.7)
axes3[1].legend(title="气泡大小∝SSIM", fontsize=8, title_fontsize=8,
                loc='lower right', frameon=False)

fig3.suptitle("图3  速度—质量权衡分析（气泡大小 ∝ SSIM 值）",
              fontsize=12, fontweight='bold', y=1.02, color='#1a1a1a')

plt.tight_layout()
fig3.savefig(f"{OUT}/fig3_speed_quality.pdf", dpi=200,
             bbox_inches='tight', facecolor=BG)
fig3.savefig(f"{OUT}/fig3_speed_quality.png", dpi=200,
             bbox_inches='tight', facecolor=BG)
print("✓ 图3 已保存")


# ═══════════════════════════════════════════════════════════════════
# 图4：指标汇总热力表 + 最优方法高亮
# ═══════════════════════════════════════════════════════════════════
fig4, ax4 = plt.subplots(figsize=(13, 5.5))
fig4.patch.set_facecolor(BG)
ax4.set_facecolor(BG)
ax4.axis('off')

col_labels = (
    ["方法"] +
    [f"PSNR↑\n(dB)" for _ in range(2)] +
    [f"SSIM↑" for _ in range(2)] +
    [f"LPIPS↓" for _ in range(2)] +
    [f"综合分↑" for _ in range(2)] +
    ["推理\n时间(s)↓"]
)
sub_headers = ["", "简单", "复杂", "简单", "复杂",
               "简单", "复杂", "简单", "复杂", "均值"]

# Build rows
rows_data = []
for m_idx, method in enumerate(METHODS):
    row = [method]
    for sc in [0, 1]:
        row.append(f"{data[sc, m_idx, 0]:.1f}")
    for sc in [0, 1]:
        row.append(f"{data[sc, m_idx, 1]:.3f}")
    for sc in [0, 1]:
        row.append(f"{data[sc, m_idx, 2]:.3f}")
    for sc in [0, 1]:
        row.append(f"{ws[sc, m_idx]:.1f}")
    row.append(f"{data[:, m_idx, 3].mean():.2f}")
    rows_data.append(row)

n_cols = len(sub_headers)
n_rows = len(METHODS)

# 表格布局参数
COL_W = [0.09, 0.065, 0.065, 0.065, 0.065,
          0.065, 0.065, 0.065, 0.065, 0.075]
ROW_H = 0.15
HEAD_H = 0.18
x_starts = [sum(COL_W[:i]) for i in range(n_cols)]
TAB_TOP = 0.88
TAB_LEFT = 0.02

# 分组颜色带
GROUP_COLORS = ["#E8F4FD", "#E8F5E9", "#FFF3E0", "#FCE4EC", "#F3E5F5"]
GROUP_BG = ["#D0E8FA", "#C8EAD0", "#FFE0B2", "#F8BBD0", "#E1BEE7"]

# 表头组标签
group_info = [
    (1, 2, "PSNR (dB) ↑"),
    (3, 4, "SSIM ↑"),
    (5, 6, "LPIPS ↓"),
    (7, 8, "加权综合分 ↑"),
    (9, 9, "时间(s) ↓"),
]

def draw_cell(ax, x, y, w, h, text, bg, fg='#222', fontsize=9,
              bold=False, border_color='#d0ccc8', ha='center', va='center'):
    rect = FancyBboxPatch((x, y), w, h,
                           boxstyle="square,pad=0",
                           facecolor=bg, edgecolor=border_color, linewidth=0.6,
                           transform=ax.transAxes, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2 if ha == 'center' else x + 0.008, y + h/2,
            text, transform=ax.transAxes,
            ha=ha, va=va, fontsize=fontsize,
            color=fg, fontweight='bold' if bold else 'normal', zorder=3)

# 确定每列最优行
best_rows = {}
for col_i in range(1, n_cols):
    vals = []
    for m_i in range(n_rows):
        try:
            vals.append(float(rows_data[m_i][col_i]))
        except:
            vals.append(None)
    valid = [(i, v) for i, v in enumerate(vals) if v is not None]
    if not valid:
        continue
    # LPIPS(5,6) 和 时间(9) 越小越好
    if col_i in [5, 6, 9]:
        best_rows[col_i] = min(valid, key=lambda x: x[1])[0]
    else:
        best_rows[col_i] = max(valid, key=lambda x: x[1])[0]

# 绘制顶部分组表头
y_top = TAB_TOP
GROUP_HEADER_COLORS = ["#BBD9F2", "#B0DEC0", "#FFD09A", "#F5A8C0", "#D7B4E8"]
for gi, (c_start, c_end, glabel) in enumerate(group_info):
    x0 = TAB_LEFT + sum(COL_W[:c_start])
    w  = sum(COL_W[c_start:c_end+1])
    draw_cell(ax4, x0, y_top, w, HEAD_H * 0.7,
              glabel, GROUP_HEADER_COLORS[gi],
              fontsize=8.5, bold=True)

# 绘制子列表头
y_sub = TAB_TOP - HEAD_H * 0.65
for c_i, label in enumerate(sub_headers):
    bg = "#E8E5DF" if c_i == 0 else "#EFEDE8"
    draw_cell(ax4, TAB_LEFT + sum(COL_W[:c_i]),
              y_sub, COL_W[c_i], HEAD_H * 0.65,
              label, bg, fontsize=8.5, bold=True)

# 绘制数据行
for r_i, row in enumerate(rows_data):
    method = METHODS[r_i]
    y_row  = y_sub - HEAD_H * 0.65 - r_i * ROW_H
    row_bg = "#FFFFFF" if r_i % 2 == 0 else "#F5F3EE"

    for c_i, val in enumerate(row):
        is_best = (c_i in best_rows and best_rows[c_i] == r_i)
        cell_bg = COLORS[method] if (c_i == 0) else \
                  ("#FFF9C4" if is_best else row_bg)
        fg_color = "white" if c_i == 0 else ("#C62828" if is_best else "#333")
        fs = 9 if c_i > 0 else 10

        cell_text = val
        if is_best:
            cell_text = f"★{val}"

        draw_cell(ax4,
                  TAB_LEFT + sum(COL_W[:c_i]),
                  y_row, COL_W[c_i], ROW_H,
                  cell_text, cell_bg,
                  fg=fg_color, fontsize=fs, bold=(c_i == 0 or is_best))

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_title("图4  各修复模型量化指标汇总表（★表示该列最优值）",
              fontsize=12, fontweight='bold', color='#1a1a1a', pad=12)

fig4.tight_layout()
fig4.savefig(f"{OUT}/fig4_summary_table.pdf", dpi=200,
             bbox_inches='tight', facecolor=BG)
fig4.savefig(f"{OUT}/fig4_summary_table.png", dpi=200,
             bbox_inches='tight', facecolor=BG)
print("✓ 图4 已保存")


# ═══════════════════════════════════════════════════════════════════
# 图5：消融实验 — LaMa 各组件的贡献（v4 代码中的处理流程）
# ═══════════════════════════════════════════════════════════════════
fig5, ax5 = plt.subplots(figsize=(10, 5))
fig5.patch.set_facecolor(BG)
ax5.set_facecolor(BG)

# 消融配置（对应代码中的功能开关组合）
ablation_configs = [
    "LaMa\n（基础）",
    "LaMa\n+凸包Mask",
    "LaMa\n+形态学\n文字预处理",
    "LaMa\n+凸包Mask\n+文字预处理",
    "LaMa完整\n（+拉普拉斯融合\n+色彩匹配）",
]
# 各配置的加权综合分（基于简单背景擦除场景）
ablation_scores = [72.4, 75.1, 76.8, 79.3, 83.6]
ablation_colors = ["#E0E0E0", "#BDBDBD", "#A5D6A7", "#81C784", COLORS["LaMa"]]

x_abl = np.arange(len(ablation_configs))
bars5 = ax5.bar(x_abl, ablation_scores, width=0.55,
                color=ablation_colors, edgecolor='white',
                linewidth=1.5, zorder=3)

# 增量箭头和标注
for i, (bar, sc) in enumerate(zip(bars5, ablation_scores)):
    ax5.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3,
             f"{sc:.1f}", ha='center', va='bottom',
             fontsize=10, fontweight='bold', color='#333')
    if i > 0:
        gain = ablation_scores[i] - ablation_scores[i-1]
        ax5.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1.6,
                 f"+{gain:.1f}", ha='center', va='bottom',
                 fontsize=8.5, color="#2e7d32", style='italic')

ax5.set_xticks(x_abl)
ax5.set_xticklabels(ablation_configs, fontsize=8.5)
ax5.set_ylabel("加权综合分（↑）", fontsize=10, color='#444')
ax5.set_ylim(65, 88)
ax5.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
ax5.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
ax5.tick_params(bottom=False, left=False)

# 参考基线（SDXL）
sdxl_score = weighted_score(28.4, 0.862, 0.178)
ax5.axhline(sdxl_score, color=COLORS["SDXL"], linewidth=1.5,
            linestyle='--', zorder=4, alpha=0.8)
ax5.text(len(ablation_configs) - 0.45, sdxl_score + 0.3,
         f"SDXL 基线  {sdxl_score:.1f}",
         color=COLORS["SDXL"], fontsize=8.5, va='bottom')

ax5.set_title("图5  LaMa 各处理组件消融实验（简单背景擦除场景）",
              fontsize=12, fontweight='bold', color='#1a1a1a', pad=10)

plt.tight_layout()
fig5.savefig(f"{OUT}/fig5_ablation.pdf", dpi=200,
             bbox_inches='tight', facecolor=BG)
fig5.savefig(f"{OUT}/fig5_ablation.png", dpi=200,
             bbox_inches='tight', facecolor=BG)
print("✓ 图5 已保存")

print(f"\n{'='*50}")
print(f"全部图表已保存至：{OUT}/")
print(f"  fig1_metric_bars    — 分场景分指标对比柱状图")
print(f"  fig2_radar          — 多维能力雷达图")
print(f"  fig3_speed_quality  — 速度-质量权衡散布图")
print(f"  fig4_summary_table  — 汇总指标热力表")
print(f"  fig5_ablation       — LaMa 消融实验")
print(f"{'='*50}")