"""
消融实验：Grounded-SAM Inpainting 全功能模块有效性验证
========================================================
基础骨干：Grounded-SAM + Stable Diffusion 1.5
待验证模块（11 个）：

  [v2] M1:DSP Mask优化  M2:ControlNet  M3:DSP线条增强  M4:拉普拉斯融合
  [v3] M5:凸包填充  M6:两阶段Inpainting  M7:删除Prompt  M8:CN屏蔽Mask  M9:色彩匹配
  [v4] M10:LaMa后端  M11:形态学文字预处理

评价指标权重（与 v4 原始代码保持一致，按任务类型区分）：
  object_removal     → PSNR×0.10 + SSIM×0.10 + LPIPS×0.40 + CLIP×0.40
  object_replacement → PSNR×0.25 + SSIM×0.25 + LPIPS×0.25 + CLIP×0.25

实验策略：
  组A（9个）  逐步累加：从 Baseline 依次叠加每个模块
  组B（5个）  ControlNet 专项：切换 SDXL 骨干，隔离 M2/M8
  组C（10个） 逐个消融：从全模型依次移除单个模块
  组D（8个）  关键组合：验证重要模块协同效果

汇总输出：
  ablation_report.json
  ablation_summary_removal.csv
  ablation_summary_replacement.csv
  ablation_summary_all.csv

运行示例：
  python ablation_study.py --run_group incremental leave_one_out key_combo
  python ablation_study.py --configs_subset A00_baseline C00_full --no_metrics
"""

import os, sys, json, csv, copy, argparse, traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# ── BERT 本地路径 ─────────────────────────────────────────────────────────────
BERT_PATH = "/root/autodl-tmp/Grounded_sam/bert-base-uncased"
if os.path.exists(BERT_PATH):
    os.environ['TRANSFORMERS_CACHE'] = os.path.dirname(BERT_PATH)
    os.environ['HF_HOME']            = os.path.dirname(BERT_PATH)

# ── 核心导入 ──────────────────────────────────────────────────────────────────
try:
    from grounded_sam_inpainting import (
        load_image, load_model, get_grounding_output,
        filter_boxes_nms, resize_with_padding, restore_from_padding,
        show_mask, show_box,
    )
    import torch, cv2
    import numpy as np
    from PIL import Image
    from segment_anything import SamPredictor, build_sam
    from diffusers import StableDiffusionInpaintPipeline
    from diffusers import (
        StableDiffusionXLInpaintPipeline,
        ControlNetModel,
        StableDiffusionXLControlNetInpaintPipeline,
    )
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.metrics import peak_signal_noise_ratio as psnr_func
    from skimage.metrics import structural_similarity  as ssim_func
    import lpips, open_clip
except ImportError as e:
    print(f"[FATAL] 导入失败: {e}")
    sys.exit(1)

# ── IOPaint / LaMa 可选 ───────────────────────────────────────────────────────
try:
    from iopaint.model_manager import ModelManager
    from iopaint.schema import InpaintRequest, HDStrategy, LDMSampler
    IOPAINT_AVAILABLE = True
    print("✓ IOPaint (LaMa) 可用")
except ImportError:
    IOPAINT_AVAILABLE = False
    print("⚠ IOPaint 未安装，M10/M11 相关实验将自动跳过")

print("✓ 形态学文字预处理就绪\n")

# =============================================================================
# ★ 路径配置（绝对路径）
# =============================================================================
_BASE = "/root/autodl-tmp/Grounded_sam/Grounded-Segment-Anything"

PATH_CONFIG: Dict[str, str] = {
    "grounding_dino_config":
        f"{_BASE}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "grounded_checkpoint":
        f"{_BASE}/weights/groundingdino_swint_ogc.pth",
    "sam_checkpoint":
        f"{_BASE}/weights/sam_vit_h_4b8939.pth",
    "image_root":
        f"{_BASE}/dataset",
    "prompts_config":
        f"{_BASE}/prompts_config.json",
    "sd15_model_path":
        "/root/autodl-tmp/models/stable-diffusion-v1-5-inpainting",
    "sdxl_model_path":
        "/root/autodl-tmp/models/sdxl-inpainting",
    "controlnet_model_path":
        "/root/autodl-tmp/models/controlnet-canny-sdxl",
    "output_base_dir":
        f"{_BASE}/ablation_outputs",
    "device": "cuda",
}

# =============================================================================
# 任务类型 → 权重键 映射
# prompts_config 里使用 "object_removal" / "object_replacement"
# TASK_WEIGHTS 键与 v4 原始 Evaluator 保持一致："removal" / "replace"
# =============================================================================
TASK_WEIGHT_KEY = {
    "object_removal":     "removal",
    "object_replacement": "replace",
}

# 与 v4 原始 Evaluator.TASK_WEIGHTS 完全一致
TASK_WEIGHTS = {
    "removal": {"PSNR": 0.10, "SSIM": 0.10, "LPIPS": 0.40, "CLIP_Score": 0.40},
    "replace": {"PSNR": 0.25, "SSIM": 0.25, "LPIPS": 0.25, "CLIP_Score": 0.25},
}

# =============================================================================
# 参数默认值
# =============================================================================
PARAM_DEFAULTS: Dict[str, Any] = {
    "box_threshold":            0.3,
    "text_threshold":           0.25,
    "nms_iou_threshold":        0.5,
    "inpaint_steps":            50,
    "inpaint_guidance_scale":   7.5,
    "inpaint_strength":         0.99,
    "inpaint_negative_prompt":
        "blurry, bad quality, distorted, artifacts, ugly, low resolution",
    # v2 模块
    "use_dsp_mask":             False,
    "dsp_morph_close_ksize":    21,
    "dsp_morph_open_ksize":     7,
    "dsp_feather_sigma":        21,
    "dsp_feather_strength":     1.0,
    "use_controlnet":           False,
    "controlnet_condition_type":"canny",
    "controlnet_scale":         0.5,
    "use_dsp_enhance":          False,
    "dsp_bilateral_d":          9,
    "dsp_bilateral_sigma":      75,
    "dsp_unsharp_strength":     0.5,
    "use_lap_blend":            False,
    "lap_blend_levels":         6,
    # v3 模块
    "use_convex_hull":          False,
    "use_two_stage":            False,
    "stage1_steps":             30,
    "stage1_strength":          0.65,
    "stage1_guidance":          9.0,
    "stage2_steps":             20,
    "stage2_strength":          0.35,
    "stage2_guidance":          7.5,
    "use_removal_prompt":       False,
    "controlnet_mask_shield":   False,
    "use_color_match":          False,
    "color_match_border_radius":30,
    "color_match_blend_radius": 15,
    # v4 模块
    "use_lama":                 False,
    "use_ocr_preprocess":       False,
    "ocr_confidence_threshold": 0.5,
    "ocr_fill_color":           (255, 255, 255),
    # 骨干
    "backbone":                 "sd15",
}

# =============================================================================
# 消融实验配置表（32 个，4 组）
# task_scope: "all" / "object_removal" / "object_replacement"
#   仅 task_scope 内的测试样本才参与该实验（避免 removal-only 模块在
#   replacement 样本上产生无意义结果，污染汇总指标）
# =============================================================================
ABLATION_CONFIGS: Dict[str, Dict] = {

    # ── 组 A：逐步累加 ────────────────────────────────────────────────────────
    "A00_baseline": {
        "name": "Baseline (SD1.5 only)",
        "group": "incremental", "task_scope": "all",
        "desc": "Grounded-SAM + SD1.5，全部改进关闭",
    },
    "A01_M1": {
        "name": "+M1 DSP Mask",
        "group": "incremental", "task_scope": "all",
        "desc": "叠加 DSP Mask 优化（形态学修补+羽化）",
        "use_dsp_mask": True,
    },
    "A02_M1M3": {
        "name": "+M1+M3 DSP Enhance",
        "group": "incremental", "task_scope": "all",
        "desc": "叠加 DSP 线条增强（仅供 SAM）",
        "use_dsp_mask": True, "use_dsp_enhance": True,
    },
    "A03_M1M3M4": {
        "name": "+M1+M3+M4 LapBlend",
        "group": "incremental", "task_scope": "all",
        "desc": "叠加拉普拉斯金字塔融合",
        "use_dsp_mask": True, "use_dsp_enhance": True, "use_lap_blend": True,
    },
    "A04_M1M3M4M5": {
        "name": "+M1+M3+M4+M5 ConvexHull",
        "group": "incremental", "task_scope": "all",
        "desc": "叠加凸包填充（removal 生效）",
        "use_dsp_mask": True, "use_dsp_enhance": True,
        "use_lap_blend": True, "use_convex_hull": True,
    },
    "A05_M1M3M4M5M7": {
        "name": "+M1+M3+M4+M5+M7 RmPrompt",
        "group": "incremental", "task_scope": "object_removal",
        "desc": "叠加删除类专用 Prompt（仅对 removal 样本测试）",
        "use_dsp_mask": True, "use_dsp_enhance": True,
        "use_lap_blend": True, "use_convex_hull": True,
        "use_removal_prompt": True,
    },
    "A06_M1M3M4M5M7M9": {
        "name": "+M1+M3+M4+M5+M7+M9 ColorMatch",
        "group": "incremental", "task_scope": "object_removal",
        "desc": "叠加边缘色彩匹配后处理",
        "use_dsp_mask": True, "use_dsp_enhance": True,
        "use_lap_blend": True, "use_convex_hull": True,
        "use_removal_prompt": True, "use_color_match": True,
    },
    "A07_M1M3M4M5M7M9M10": {
        "name": "+M1+M3+M4+M5+M7+M9+M10 LaMa",
        "group": "incremental", "task_scope": "object_removal",
        "desc": "叠加 LaMa 后端替代 SD1.5（removal 专用）",
        "use_dsp_mask": True, "use_dsp_enhance": True,
        "use_lap_blend": True, "use_convex_hull": True,
        "use_removal_prompt": True, "use_color_match": True,
        "use_lama": True,
    },
    "A08_full_no_cn": {
        "name": "Full (no ControlNet)",
        "group": "incremental", "task_scope": "all",
        "desc": "M1+M3+M4+M5+M6+M7+M9+M10+M11 全开，不含 M2/M8",
        "use_dsp_mask": True, "use_dsp_enhance": True,
        "use_lap_blend": True, "use_convex_hull": True,
        "use_two_stage": True, "use_removal_prompt": True,
        "use_color_match": True, "use_lama": True, "use_ocr_preprocess": True,
    },

    # ── 组 B：ControlNet 专项（SDXL 骨干）────────────────────────────────────
    "B00_sdxl": {
        "name": "SDXL Baseline",
        "group": "controlnet", "task_scope": "all",
        "desc": "SDXL 管线，全部模块关闭",
        "backbone": "sdxl",
    },
    "B01_M2": {
        "name": "+M2 ControlNet",
        "group": "controlnet", "task_scope": "all",
        "desc": "SDXL + ControlNet Canny",
        "backbone": "sdxl", "use_controlnet": True,
    },
    "B02_M2M8": {
        "name": "+M2+M8 CN+MaskShield",
        "group": "controlnet", "task_scope": "all",
        "desc": "ControlNet + 条件图屏蔽 Mask 区域",
        "backbone": "sdxl", "use_controlnet": True, "controlnet_mask_shield": True,
    },
    "B03_M2M8M1M3M4": {
        "name": "+M2+M8+M1+M3+M4",
        "group": "controlnet", "task_scope": "all",
        "desc": "ControlNet 组合 + DSP 全套",
        "backbone": "sdxl", "use_controlnet": True,
        "controlnet_mask_shield": True,
        "use_dsp_mask": True, "use_dsp_enhance": True, "use_lap_blend": True,
    },
    "B04_M2M8M1M3M4M6": {
        "name": "+M2+M8+M1+M3+M4+M6",
        "group": "controlnet", "task_scope": "all",
        "desc": "B03 + 两阶段 Inpainting",
        "backbone": "sdxl", "use_controlnet": True,
        "controlnet_mask_shield": True,
        "use_dsp_mask": True, "use_dsp_enhance": True,
        "use_lap_blend": True, "use_two_stage": True,
    },

    # ── 组 C：逐个消融（Leave-One-Out）────────────────────────────────────────
    "C00_full": {
        "name": "Full Model (all ON)",
        "group": "leave_one_out", "task_scope": "all",
        "desc": "全部模块开启（非 ControlNet 分支）",
        "use_dsp_mask": True, "use_dsp_enhance": True,
        "use_lap_blend": True, "use_convex_hull": True,
        "use_two_stage": True, "use_removal_prompt": True,
        "use_color_match": True, "use_lama": True, "use_ocr_preprocess": True,
    },
    "C01_noM1": {
        "name": "Full - M1",
        "group": "leave_one_out", "task_scope": "all",
        "desc": "移除 DSP Mask 优化",
        "use_dsp_mask": False,
        "use_dsp_enhance": True, "use_lap_blend": True, "use_convex_hull": True,
        "use_two_stage": True, "use_removal_prompt": True,
        "use_color_match": True, "use_lama": True, "use_ocr_preprocess": True,
    },
    "C02_noM3": {
        "name": "Full - M3",
        "group": "leave_one_out", "task_scope": "all",
        "desc": "移除 DSP 线条增强",
        "use_dsp_mask": True, "use_dsp_enhance": False,
        "use_lap_blend": True, "use_convex_hull": True,
        "use_two_stage": True, "use_removal_prompt": True,
        "use_color_match": True, "use_lama": True, "use_ocr_preprocess": True,
    },
    "C03_noM4": {
        "name": "Full - M4",
        "group": "leave_one_out", "task_scope": "all",
        "desc": "移除拉普拉斯金字塔融合",
        "use_dsp_mask": True, "use_dsp_enhance": True,
        "use_lap_blend": False, "use_convex_hull": True,
        "use_two_stage": True, "use_removal_prompt": True,
        "use_color_match": True, "use_lama": True, "use_ocr_preprocess": True,
    },
    "C04_noM5": {
        "name": "Full - M5",
        "group": "leave_one_out", "task_scope": "object_removal",
        "desc": "移除凸包填充（仅对 removal 样本有意义）",
        "use_dsp_mask": True, "use_dsp_enhance": True,
        "use_lap_blend": True, "use_convex_hull": False,
        "use_two_stage": True, "use_removal_prompt": True,
        "use_color_match": True, "use_lama": True, "use_ocr_preprocess": True,
    },
    "C05_noM6": {
        "name": "Full - M6",
        "group": "leave_one_out", "task_scope": "all",
        "desc": "移除两阶段 Inpainting",
        "use_dsp_mask": True, "use_dsp_enhance": True,
        "use_lap_blend": True, "use_convex_hull": True,
        "use_two_stage": False, "use_removal_prompt": True,
        "use_color_match": True, "use_lama": True, "use_ocr_preprocess": True,
    },
    "C06_noM7": {
        "name": "Full - M7",
        "group": "leave_one_out", "task_scope": "object_removal",
        "desc": "移除删除类专用 Prompt（仅 removal 可见差异）",
        "use_dsp_mask": True, "use_dsp_enhance": True,
        "use_lap_blend": True, "use_convex_hull": True,
        "use_two_stage": True, "use_removal_prompt": False,
        "use_color_match": True, "use_lama": True, "use_ocr_preprocess": True,
    },
    "C07_noM9": {
        "name": "Full - M9",
        "group": "leave_one_out", "task_scope": "object_removal",
        "desc": "移除边缘色彩匹配（仅 removal 可见差异）",
        "use_dsp_mask": True, "use_dsp_enhance": True,
        "use_lap_blend": True, "use_convex_hull": True,
        "use_two_stage": True, "use_removal_prompt": True,
        "use_color_match": False, "use_lama": True, "use_ocr_preprocess": True,
    },
    "C08_noM10": {
        "name": "Full - M10 (SD1.5 fallback)",
        "group": "leave_one_out", "task_scope": "all",
        "desc": "移除 LaMa，removal 回退到 SD1.5",
        "use_dsp_mask": True, "use_dsp_enhance": True,
        "use_lap_blend": True, "use_convex_hull": True,
        "use_two_stage": True, "use_removal_prompt": True,
        "use_color_match": True, "use_lama": False, "use_ocr_preprocess": False,
    },
    "C09_noM11": {
        "name": "Full - M11",
        "group": "leave_one_out", "task_scope": "object_removal",
        "desc": "移除形态学文字预处理（仅 removal 可见差异）",
        "use_dsp_mask": True, "use_dsp_enhance": True,
        "use_lap_blend": True, "use_convex_hull": True,
        "use_two_stage": True, "use_removal_prompt": True,
        "use_color_match": True, "use_lama": True, "use_ocr_preprocess": False,
    },

    # ── 组 D：关键组合对比 ────────────────────────────────────────────────────
    "D01_lama_only": {
        "name": "LaMa only",
        "group": "key_combo", "task_scope": "object_removal",
        "desc": "仅 LaMa，无任何后处理，验证 M10 原始效果",
        "use_lama": True,
    },
    "D02_lama_ocr": {
        "name": "LaMa + OCR",
        "group": "key_combo", "task_scope": "object_removal",
        "desc": "M10+M11 协同：LaMa + 形态学文字预处理",
        "use_lama": True, "use_ocr_preprocess": True,
    },
    "D03_lama_mask_blend": {
        "name": "LaMa + Mask + Blend",
        "group": "key_combo", "task_scope": "object_removal",
        "desc": "核心融合链：M10+M1+M4",
        "use_lama": True, "use_dsp_mask": True, "use_lap_blend": True,
    },
    "D04_sd15_full_nolama": {
        "name": "SD1.5 Full (no LaMa)",
        "group": "key_combo", "task_scope": "all",
        "desc": "全模块开启但不用 LaMa，测试 SD1.5 上限",
        "use_dsp_mask": True, "use_dsp_enhance": True,
        "use_lap_blend": True, "use_convex_hull": True,
        "use_two_stage": True, "use_removal_prompt": True,
        "use_color_match": True, "use_lama": False, "use_ocr_preprocess": False,
    },
    "D05_two_stage_only": {
        "name": "Two-Stage only",
        "group": "key_combo", "task_scope": "object_replacement",
        "desc": "仅两阶段 Inpainting，隔离其对 replacement 的贡献",
        "use_two_stage": True,
    },
    "D06_color_match_only": {
        "name": "ColorMatch only",
        "group": "key_combo", "task_scope": "object_removal",
        "desc": "仅色彩匹配，隔离 M9 独立贡献",
        "use_color_match": True,
    },
    "D07_mask_chain": {
        "name": "Mask Chain (M1+M3+M5)",
        "group": "key_combo", "task_scope": "all",
        "desc": "完整 Mask 处理链协同验证",
        "use_dsp_mask": True, "use_dsp_enhance": True, "use_convex_hull": True,
    },
    "D08_post_chain": {
        "name": "Post Chain (M4+M9)",
        "group": "key_combo", "task_scope": "object_removal",
        "desc": "后处理链（拉普拉斯融合+色彩匹配）协同验证",
        "use_lap_blend": True, "use_color_match": True,
    },
}


# =============================================================================
# 功能模块实现（从 v4 原始代码完整复制）
# =============================================================================

# ── LaMa (M10) ────────────────────────────────────────────────────────────────
_lama_manager = None

def get_lama_model(device="cuda"):
    global _lama_manager
    if _lama_manager is None:
        if not IOPAINT_AVAILABLE:
            raise RuntimeError("IOPaint 未安装")
        print("  [M10] 加载 LaMa 模型...")
        _lama_manager = ModelManager(name="lama", device=device)
        print("  ✓ LaMa 加载完成")
    return _lama_manager

def lama_inpaint(image_np, mask_np, device="cuda"):
    model = get_lama_model(device)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    ms = mask_np if mask_np.ndim == 2 else mask_np[:, :, 0]
    _, mb = cv2.threshold(ms, 127, 255, cv2.THRESH_BINARY)
    req = InpaintRequest(
        hd_strategy=HDStrategy.ORIGINAL,
        hd_strategy_crop_margin=32,
        hd_strategy_crop_trigger_size=800,
        hd_strategy_resize_limit=1280,
        ldm_sampler=LDMSampler.ddim,
    )
    return cv2.cvtColor(model(image_bgr, mb, req), cv2.COLOR_BGR2RGB).astype(np.uint8)


# ── 形态学文字预处理 (M11) ────────────────────────────────────────────────────
def ocr_fill_text(image_np, mask_np, fill_color=(255,255,255),
                  ocr_confidence_threshold=0.5):
    out = image_np.copy()
    mb  = (mask_np > 127).astype(np.uint8)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, dark = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    cand = cv2.bitwise_and(dark, dark, mask=mb)
    conn = cv2.dilate(cand, cv2.getStructuringElement(cv2.MORPH_RECT,(5,3)), iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(conn, 8)
    amax = max(100, int(mb.sum()) * 0.15)
    tmask = np.zeros_like(dark)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if 20 <= a <= amax:
            tmask[labels == i] = 255
    tmask = cv2.dilate(tmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
    region = (tmask > 0) & (mb > 0)
    if region.sum() > 0:
        out[region] = fill_color
    return out


# ── 凸包填充 (M5) ─────────────────────────────────────────────────────────────
def fill_convex_hull(mask_np):
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hm = np.zeros_like(mask_np)
    for cnt in contours:
        if len(cnt) >= 3:
            cv2.drawContours(hm, [cv2.convexHull(cnt)], -1, 255, -1)
    return cv2.bitwise_or(mask_np, hm)


# ── DSP Mask 优化 (M1) ────────────────────────────────────────────────────────
def dsp_optimize_mask(mask_np, morph_close_ksize=21, morph_open_ksize=7,
                      dilate_ksize=15, dilate_iters=1, feather_sigma=21,
                      feather_strength=1.0, use_convex_hull=False):
    if use_convex_hull:
        mask_np = fill_convex_hull(mask_np)
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(morph_close_ksize,morph_close_ksize))
    m  = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kc)
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(morph_open_ksize,morph_open_ksize))
    m  = cv2.morphologyEx(m, cv2.MORPH_OPEN, ko)
    m  = cv2.dilate(m, np.ones((dilate_ksize,dilate_ksize),np.uint8), iterations=dilate_iters)
    fk = feather_sigma if feather_sigma % 2 == 1 else feather_sigma + 1
    mf = cv2.GaussianBlur(m, (fk, fk), 0)
    return (cv2.addWeighted(mf, feather_strength, m, 1-feather_strength, 0)
            if feather_strength < 1.0 else mf).astype(np.uint8)


# ── DSP 线条增强 (M3) ─────────────────────────────────────────────────────────
def dsp_enhance_image(image_np, bilateral_d=9, bilateral_sigma_color=75,
                      bilateral_sigma_space=75, unsharp_strength=0.5, unsharp_sigma=3):
    bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    f   = cv2.bilateralFilter(bgr, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
    k   = unsharp_sigma if unsharp_sigma % 2 == 1 else unsharp_sigma + 1
    bl  = cv2.GaussianBlur(f, (k*2+1, k*2+1), unsharp_sigma)
    sh  = cv2.addWeighted(f, 1+unsharp_strength, bl, -unsharp_strength, 0)
    return cv2.cvtColor(np.clip(sh,0,255).astype(np.uint8), cv2.COLOR_BGR2RGB)


# ── 拉普拉斯金字塔融合 (M4) ───────────────────────────────────────────────────
def _gp(img, lv):
    p = [img.astype(np.float32)]
    for _ in range(lv-1):
        img = cv2.pyrDown(img); p.append(img.astype(np.float32))
    return p

def _lp(img, lv):
    g = _gp(img, lv); lap = []
    for i in range(lv-1):
        up = cv2.pyrUp(g[i+1], dstsize=(g[i].shape[1], g[i].shape[0]))
        lap.append(g[i].astype(np.float32) - up.astype(np.float32))
    lap.append(g[-1].astype(np.float32))
    return lap

def laplacian_pyramid_blend(orig_np, inpainted_np, mask_np, levels=6):
    mf = mask_np.astype(np.float32)/255.0
    if mf.ndim == 2: mf = np.stack([mf]*3, axis=-1)
    lo, li, gm = _lp(orig_np, levels), _lp(inpainted_np, levels), _gp(mf, levels)
    blended = [a*(1-g)+b*g for a,b,g in zip(lo,li,gm)]
    res = blended[-1]
    for i in range(levels-2, -1, -1):
        res = cv2.pyrUp(res, dstsize=(blended[i].shape[1],blended[i].shape[0])) + blended[i]
    return np.clip(res, 0, 255).astype(np.uint8)


# ── 边缘色彩匹配 (M9) ─────────────────────────────────────────────────────────
def color_match_inpainted(orig_np, inpainted_np, mask_np,
                          border_radius=30, blend_radius=15):
    result = inpainted_np.astype(np.float32)
    kb = np.ones((border_radius*2+1, border_radius*2+1), np.uint8)
    mb = (mask_np > 127).astype(np.uint8) * 255
    ring = cv2.bitwise_and(cv2.dilate(mb, kb), cv2.bitwise_not(mb))
    for c in range(3):
        bp = orig_np.astype(np.float32)[:,:,c][ring > 0]
        ip = result[:,:,c][mb > 0]
        if len(bp) < 10 or len(ip) < 10: continue
        sm, ss = ip.mean(), ip.std()
        tm, ts = bp.mean(), bp.std()
        if ss < 1e-6: continue
        scale = ts/(ss+1e-8); shift = tm - scale*sm
        corr = np.clip(result[:,:,c]*scale+shift, 0, 255)
        bw   = np.clip(cv2.distanceTransform(mb, cv2.DIST_L2, 5)/(blend_radius+1e-8), 0, 1)
        result[:,:,c] = np.where(mb>0, result[:,:,c]*(1-bw)+corr*bw, result[:,:,c])
    return np.clip(result, 0, 255).astype(np.uint8)


# ── ControlNet 条件图 (M2+M8) ─────────────────────────────────────────────────
def generate_controlnet_condition(image_np, condition_type="canny", mask_np=None):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200) if condition_type == "canny" else gray
    rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    if mask_np is not None:
        mr = cv2.resize(mask_np, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        rgb[mr > 127] = 0
    return Image.fromarray(rgb)


# ── 两阶段 Inpainting (M6) ────────────────────────────────────────────────────
def two_stage_inpaint(pipe, prompt, negative_prompt, image_padded, mask_padded,
                       stage1_steps=30, stage1_strength=0.65, stage1_guidance=9.0,
                       stage2_steps=20, stage2_strength=0.35, stage2_guidance=7.5,
                       use_controlnet=False, condition_padded=None, controlnet_scale=0.5):
    kw = dict(prompt=prompt, negative_prompt=negative_prompt, mask_image=mask_padded)
    if use_controlnet and condition_padded is not None:
        kw["control_image"] = condition_padded
        kw["controlnet_conditioning_scale"] = controlnet_scale
    s1 = pipe(**kw, image=image_padded, num_inference_steps=stage1_steps,
               guidance_scale=stage1_guidance, strength=stage1_strength).images[0]
    return pipe(**kw, image=s1, num_inference_steps=stage2_steps,
                guidance_scale=stage2_guidance, strength=stage2_strength).images[0]


# ── Prompt 增强 (M7) ──────────────────────────────────────────────────────────
_NEG_KW = (
    "speech bubble, thought bubble, text bubble, caption, subtitle, "
    "text, letter, word, font, typography, watermark, label, annotation, "
    "border, outline, stroke, shadow of text, comic panel border"
)
_POS_SFX = (
    "seamless background continuation, no text, no bubble, "
    "consistent texture, consistent lighting, photorealistic, clean surface"
)

def build_removal_prompts(prompt, neg_prompt, use_removal_prompt=False,
                           task_type="object_replacement"):
    if use_removal_prompt and task_type == "object_removal":
        return f"{prompt}, {_POS_SFX}", f"{neg_prompt}, {_NEG_KW}"
    return prompt, neg_prompt


# =============================================================================
# 评价指标类（与 v4 原始 Evaluator 完全一致，增加任务类型映射）
# =============================================================================
class Evaluator:
    def __init__(self, device):
        self.device = device
        print("初始化评价指标（LPIPS + CLIP）...")
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        self.clip_model, _, self.clip_preprocess = (
            open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k'))
        self.clip_model = self.clip_model.to(device)
        self.tokenizer  = open_clip.get_tokenizer('ViT-B-32')
        print("✓ 评价指标加载完成\n")

    def run(self, org_path, res_path, prompt, task_type="object_replacement"):
        """
        task_type: "object_removal" | "object_replacement"
        内部通过 TASK_WEIGHT_KEY 映射到 TASK_WEIGHTS 键，与 v4 原始代码一致。
        """
        try:
            io = Image.open(org_path).convert('RGB').resize((512,512))
            ir = Image.open(res_path).convert('RGB').resize((512,512))
            no, nr = np.array(io), np.array(ir)

            psnr_v = psnr_func(no, nr, data_range=255)
            ssim_v = ssim_func(no, nr, channel_axis=2, data_range=255)

            to_ = lpips.im2tensor(no).to(self.device)
            tr_ = lpips.im2tensor(nr).to(self.device)
            lpips_v = self.lpips_model(to_, tr_).item()

            ii = self.clip_preprocess(ir).unsqueeze(0).to(self.device)
            ti = self.tokenizer([prompt]).to(self.device)
            with torch.no_grad():
                imf = self.clip_model.encode_image(ii)
                txf = self.clip_model.encode_text(ti)
                imf = imf / imf.norm(dim=-1, keepdim=True)
                txf = txf / txf.norm(dim=-1, keepdim=True)
                clip_v = (imf @ txf.T).item() * 100

            # ★ 核心：通过映射表获取正确权重键，与 v4 原始 Evaluator 完全一致
            wkey    = TASK_WEIGHT_KEY.get(task_type, "replace")
            weights = TASK_WEIGHTS[wkey]

            lpips_score = max(0.0, 1.0 - lpips_v)
            psnr_norm   = min(psnr_v / 50.0, 1.0)
            wscore = (
                weights["PSNR"]       * psnr_norm  +
                weights["SSIM"]       * ssim_v      +
                weights["LPIPS"]      * lpips_score +
                weights["CLIP_Score"] * (clip_v / 100.0)
            ) * 100

            return {
                "PSNR":           round(psnr_v,  2),
                "SSIM":           round(ssim_v,  4),
                "LPIPS":          round(lpips_v, 4),
                "CLIP_Score":     round(clip_v,  2),
                "weighted_score": round(wscore,  2),
                "weight_key":     wkey,
                "weights_used":   weights,
            }
        except Exception as e:
            print(f"    ⚠ 评价失败: {e}")
            return None


# =============================================================================
# 管线缓存
# =============================================================================
_pipe_cache: Dict[str, Any] = {}

def get_pipe(backbone, use_controlnet, condition_type="canny", device="cuda"):
    key = f"{backbone}_cn{use_controlnet}_{condition_type}"
    if key in _pipe_cache:
        return _pipe_cache[key]
    print(f"\n  [Pipe] backbone={backbone}, controlnet={use_controlnet}")
    if backbone == "sd15":
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            PATH_CONFIG["sd15_model_path"], torch_dtype=torch.float16,
            local_files_only=True).to(device)
        pipe._tsize = 512
    elif backbone == "sdxl":
        if use_controlnet:
            cn = ControlNetModel.from_pretrained(
                PATH_CONFIG["controlnet_model_path"],
                torch_dtype=torch.float16, local_files_only=True)
            pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                PATH_CONFIG["sdxl_model_path"], controlnet=cn,
                torch_dtype=torch.float16, local_files_only=True).to(device)
        else:
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                PATH_CONFIG["sdxl_model_path"], torch_dtype=torch.float16,
                local_files_only=True).to(device)
        pipe._tsize = 1024
    else:
        raise ValueError(f"未知 backbone: {backbone}")
    print("  ✓ 管线加载完成")
    _pipe_cache[key] = pipe
    return pipe


# =============================================================================
# 单样本推理
# =============================================================================
def run_ablation_test(image_path, det_prompt, inpaint_prompt, description,
                       task_type, dino_model, sam_checkpoint, params,
                       output_dir, evaluator=None, device="cuda"):
    try:
        p     = params
        is_rm = (task_type == "object_removal")
        eff_convex = p["use_convex_hull"]    and is_rm
        eff_color  = p["use_color_match"]    and is_rm
        eff_lama   = p["use_lama"]           and is_rm and IOPAINT_AVAILABLE
        eff_ocr    = p["use_ocr_preprocess"] and eff_lama
        eff_mode   = "merge" if is_rm else "first"

        os.makedirs(output_dir, exist_ok=True)

        # [1] 加载图片
        image_pil, image_tensor = load_image(image_path)
        raw_path = os.path.join(output_dir, "raw_image.jpg")
        image_pil.save(raw_path)

        # [2] DINO 检测
        boxes, phrases, scores = get_grounding_output(
            dino_model, image_tensor, det_prompt,
            p["box_threshold"], p["text_threshold"], device=device)
        boxes, phrases, scores = filter_boxes_nms(
            boxes, phrases, scores, iou_threshold=p["nms_iou_threshold"])
        if len(boxes) == 0:
            return {"status": "warning", "message": "未检测到目标",
                    "image_path": image_path, "description": description}

        # [3] SAM（M3 增强图仅供 SAM）
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
        image_cv  = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        sam_img   = dsp_enhance_image(
            image_cv,
            bilateral_d=p["dsp_bilateral_d"],
            bilateral_sigma_color=p["dsp_bilateral_sigma"],
            bilateral_sigma_space=p["dsp_bilateral_sigma"],
            unsharp_strength=p["dsp_unsharp_strength"],
        ) if p["use_dsp_enhance"] else image_cv
        predictor.set_image(sam_img)

        H, W = image_pil.size[1], image_pil.size[0]
        for i in range(boxes.size(0)):
            boxes[i] = boxes[i] * torch.Tensor([W, H, W, H])
            boxes[i][:2] -= boxes[i][2:] / 2
            boxes[i][2:] += boxes[i][:2]
        boxes   = boxes.cpu()
        tb      = predictor.transform.apply_boxes_torch(boxes, image_cv.shape[:2]).to(device)
        masks, _, _ = predictor.predict_torch(
            point_coords=None, point_labels=None, boxes=tb, multimask_output=False)

        # [4] Mask（M1 含 M5）
        raw_mask = (torch.any(masks, dim=0)[0] if eff_mode == "merge"
                    else masks[0][0]).cpu().numpy().astype(np.uint8) * 255
        if p["use_dsp_mask"]:
            merged_mask = dsp_optimize_mask(
                raw_mask,
                morph_close_ksize=p["dsp_morph_close_ksize"],
                morph_open_ksize=p["dsp_morph_open_ksize"],
                feather_sigma=p["dsp_feather_sigma"],
                feather_strength=p["dsp_feather_strength"],
                use_convex_hull=eff_convex,
            )
        else:
            merged_mask = cv2.GaussianBlur(
                cv2.dilate(raw_mask, np.ones((15,15),np.uint8), iterations=1), (21,21), 0)

        # M11：文字预处理
        img_inp = (ocr_fill_text(image_cv, merged_mask, fill_color=p["ocr_fill_color"])
                   if eff_ocr else image_cv.copy())
        pil_inp = Image.fromarray(img_inp)
        msk_pil = Image.fromarray(merged_mask)
        orig_sz = pil_inp.size

        # M7：Prompt 增强
        enh_p, enh_n = build_removal_prompts(
            inpaint_prompt, p["inpaint_negative_prompt"],
            use_removal_prompt=p["use_removal_prompt"], task_type=task_type)

        # [5] Inpainting
        if eff_lama:
            inp_np  = lama_inpaint(np.array(pil_inp), merged_mask, device=device)
            inp_pil = Image.fromarray(inp_np)
        else:
            pipe  = get_pipe(p["backbone"], p["use_controlnet"],
                              p["controlnet_condition_type"], device)
            tsize = getattr(pipe, "_tsize", 512)
            ip, nwh, off = resize_with_padding(pil_inp, target_size=tsize)
            mp, _,   _   = resize_with_padding(msk_pil.convert("RGB"), target_size=tsize)
            mp = mp.convert("L")
            cpd = None
            if p["use_controlnet"]:
                cond = generate_controlnet_condition(
                    np.array(pil_inp), condition_type=p["controlnet_condition_type"],
                    mask_np=merged_mask if p["controlnet_mask_shield"] else None)
                cpd, _, _ = resize_with_padding(cond, target_size=tsize)
            if p["use_two_stage"]:
                inp_pil = two_stage_inpaint(
                    pipe=pipe, prompt=enh_p, negative_prompt=enh_n,
                    image_padded=ip, mask_padded=mp,
                    stage1_steps=p["stage1_steps"], stage1_strength=p["stage1_strength"],
                    stage1_guidance=p["stage1_guidance"], stage2_steps=p["stage2_steps"],
                    stage2_strength=p["stage2_strength"], stage2_guidance=p["stage2_guidance"],
                    use_controlnet=p["use_controlnet"], condition_padded=cpd,
                    controlnet_scale=p["controlnet_scale"])
            else:
                kw = dict(prompt=enh_p, negative_prompt=enh_n,
                           image=ip, mask_image=mp,
                           num_inference_steps=p["inpaint_steps"],
                           guidance_scale=p["inpaint_guidance_scale"],
                           strength=p["inpaint_strength"])
                if p["use_controlnet"] and cpd is not None:
                    kw["control_image"] = cpd
                    kw["controlnet_conditioning_scale"] = p["controlnet_scale"]
                inp_pil = pipe(**kw).images[0]
            inp_pil = restore_from_padding(inp_pil, orig_sz, nwh, off, target_size=tsize)
            inp_np  = np.array(inp_pil)

        # [6] 后处理
        base = img_inp
        mr   = cv2.resize(merged_mask, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
        if p["use_lap_blend"]:
            ir = cv2.resize(inp_np, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
            blended = laplacian_pyramid_blend(base, ir, mr, levels=p["lap_blend_levels"])
        else:
            blended = cv2.resize(inp_np, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
        if eff_color:
            blended = color_match_inpainted(base, blended, mr,
                                             border_radius=p["color_match_border_radius"],
                                             blend_radius=p["color_match_blend_radius"])

        # [7] 保存
        res_path = os.path.join(output_dir, "result.jpg")
        Image.fromarray(blended).save(res_path)

        # [8] 评价指标（分任务类型加权）
        metrics = None
        if evaluator is not None:
            metrics = evaluator.run(raw_path, res_path, enh_p, task_type=task_type)

        return {
            "status": "success", "image_path": image_path,
            "description": description, "task_type": task_type,
            "output_dir": output_dir, "n_detected": len(boxes),
            "metrics": metrics,
        }
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "image_path": image_path,
                "description": description, "error": str(e)}


# =============================================================================
# 工具函数
# =============================================================================
def load_test_configs(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def merge_params(ablation_override):
    p = copy.deepcopy(PARAM_DEFAULTS)
    for k, v in ablation_override.items():
        if k not in ("name", "group", "desc", "task_scope"):
            p[k] = v
    return p

def check_feasibility(params):
    if params["use_lama"] and not IOPAINT_AVAILABLE:
        return "IOPaint 未安装，跳过"
    if params["use_controlnet"] and params["backbone"] != "sdxl":
        return "ControlNet 需要 SDXL 骨干"
    return None

def compute_avg(records):
    valid = [r["metrics"] for r in records
             if r.get("status") == "success" and r.get("metrics")]
    if not valid:
        return None
    keys = ["PSNR", "SSIM", "LPIPS", "CLIP_Score", "weighted_score"]
    return {k: round(float(np.mean([m[k] for m in valid])), 4) for k in keys}

def _safe(obj):
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    raise TypeError(type(obj))


# =============================================================================
# 打印分任务汇总表
# =============================================================================
def print_task_table(title, rows):
    """rows: [(exp_id, name, avg_or_None, n_success, skipped)]"""
    W   = [34, 7, 7, 7, 7, 9, 5]
    HDR = ["实验", "PSNR↑", "SSIM↑", "LPIPS↓", "CLIP↑", "W.Score↑", "N"]
    sep = "+" + "+".join("-"*w for w in W) + "+"
    fmt = "|" + "|".join(f"{{:<{w}}}" for w in W) + "|"
    print(f"\n  ── {title}")
    print(sep); print(fmt.format(*HDR)); print(sep)
    for exp_id, name, avg, n, skipped in rows:
        tag = name[:32]
        if skipped:
            print(fmt.format(tag, *["SKIP"]*5, str(n)))
        elif avg:
            print(fmt.format(tag,
                f"{avg['PSNR']:.2f}", f"{avg['SSIM']:.4f}",
                f"{avg['LPIPS']:.4f}", f"{avg['CLIP_Score']:.2f}",
                f"{avg['weighted_score']:.2f}", str(n)))
        else:
            print(fmt.format(tag, *["N/A"]*5, str(n)))
    print(sep)

def save_csv(rows, path):
    if not rows: return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  ✓ CSV: {path}")


# =============================================================================
# 消融实验主流程
# =============================================================================
def run_ablation(
    run_groups:     List[str]        = ("incremental", "leave_one_out", "key_combo"),
    task_filter:    str              = "all",
    configs_subset: Optional[List[str]] = None,
    enable_metrics: bool             = True,
    device:         str              = "cuda",
):
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = os.path.join(PATH_CONFIG["output_base_dir"], f"ablation_{ts}")
    os.makedirs(out_base, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"  消融实验启动  {ts}")
    print(f"  实验组: {list(run_groups)}  |  任务过滤: {task_filter}")
    print(f"  输出目录: {out_base}")
    print(f"  权重 object_removal:     PSNR×0.10  SSIM×0.10  LPIPS×0.40  CLIP×0.40")
    print(f"  权重 object_replacement: PSNR×0.25  SSIM×0.25  LPIPS×0.25  CLIP×0.25")
    print(f"{'='*80}\n")

    print("加载 Grounding DINO...")
    dino = load_model(PATH_CONFIG["grounding_dino_config"],
                       PATH_CONFIG["grounded_checkpoint"], device=device)
    print("✓ DINO 加载完成\n")

    evaluator = Evaluator(device) if enable_metrics else None

    all_tests = load_test_configs(PATH_CONFIG["prompts_config"])
    if task_filter != "all":
        all_tests = {k: v for k, v in all_tests.items()
                     if v.get("task_type", "object_replacement") == task_filter}
    print(f"✓ 测试用例: {len(all_tests)} 条\n")

    sel = [eid for eid, cfg in ABLATION_CONFIGS.items()
           if cfg["group"] in run_groups
           and (configs_subset is None or eid in configs_subset)]
    print(f"✓ 待运行实验: {len(sel)} 个\n")

    all_results: Dict[str, List[Dict]] = {}

    for ei, exp_id in enumerate(sel, 1):
        cfg    = ABLATION_CONFIGS[exp_id]
        params = merge_params(cfg)
        scope  = cfg.get("task_scope", "all")

        print(f"\n{'='*80}")
        print(f"[{ei}/{len(sel)}] {exp_id}  |  {cfg['name']}")
        print(f"  说明: {cfg['desc']}")
        print(f"  task_scope={scope}  backbone={params['backbone']}")
        on = [f"M{i}" for i, v in enumerate([
            params["use_dsp_mask"],       params["use_controlnet"],
            params["use_dsp_enhance"],    params["use_lap_blend"],
            params["use_convex_hull"],    params["use_two_stage"],
            params["use_removal_prompt"], params["controlnet_mask_shield"],
            params["use_color_match"],    params["use_lama"],
            params["use_ocr_preprocess"],
        ], start=1) if v]
        print(f"  开启: {', '.join(on) if on else '无（Baseline）'}")

        skip = check_feasibility(params)
        if skip:
            print(f"  ⚠ 跳过: {skip}")
            all_results[exp_id] = [{"status": "skipped", "reason": skip}]
            continue

        exp_results = []
        for si, (rel, cfg_t) in enumerate(all_tests.items(), 1):
            t_type   = cfg_t.get("task_type", "object_replacement")
            img_path = os.path.join(PATH_CONFIG["image_root"], rel)

            # ★ task_scope 过滤：避免 removal-only 模块在 replacement 样本上测试
            if scope != "all" and t_type != scope:
                continue

            print(f"\n  [{si}/{len(all_tests)}] {cfg_t['description']}  (task={t_type})")
            if not os.path.exists(img_path):
                print(f"    ⚠ 不存在: {img_path}")
                exp_results.append({"status": "error", "image_path": rel,
                                    "error": "文件不存在"})
                continue

            res = run_ablation_test(
                image_path=img_path,
                det_prompt=cfg_t["det_prompt"],
                inpaint_prompt=cfg_t["inpaint_prompt"],
                description=cfg_t["description"],
                task_type=t_type,
                dino_model=dino,
                sam_checkpoint=PATH_CONFIG["sam_checkpoint"],
                params=params,
                output_dir=os.path.join(out_base, exp_id, Path(rel).stem),
                evaluator=evaluator,
                device=device,
            )
            res["rel_path"] = rel; res["task_type"] = t_type
            exp_results.append(res)

            if res.get("metrics"):
                m  = res["metrics"]
                wk = m.get("weight_key", "?")
                print(f"    ✓ [{wk}] "
                      f"PSNR={m['PSNR']:.2f} SSIM={m['SSIM']:.4f} "
                      f"LPIPS={m['LPIPS']:.4f} CLIP={m['CLIP_Score']:.2f} "
                      f"W={m['weighted_score']:.2f}")

        all_results[exp_id] = exp_results

    # ── 汇总 ────────────────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("  消融实验结果汇总（分任务类型）")
    print(f"{'='*80}")

    summary = {}
    csv_rm, csv_rp, csv_all = [], [], []

    for exp_id, results in all_results.items():
        cfg   = ABLATION_CONFIGS[exp_id]
        group = cfg["group"]
        if group not in summary:
            summary[group] = {}

        skipped = any(r.get("status") == "skipped" for r in results)
        n_ok    = sum(1 for r in results if r.get("status") == "success")
        rm_recs = [r for r in results if r.get("task_type") == "object_removal"]
        rp_recs = [r for r in results if r.get("task_type") == "object_replacement"]
        avg_rm  = compute_avg(rm_recs)
        avg_rp  = compute_avg(rp_recs)
        avg_all = compute_avg(results)

        summary[group][exp_id] = {
            "name": cfg["name"], "desc": cfg["desc"],
            "skipped": skipped, "n_success": n_ok,
            "avg_removal":     avg_rm,
            "avg_replacement": avg_rp,
            "avg_all":         avg_all,
            "detail":          results,
        }

        def _row(avg, task, n):
            return {"exp_id": exp_id, "group": group, "name": cfg["name"],
                    "task": task,
                    "PSNR":           avg.get("PSNR","")          if avg else "",
                    "SSIM":           avg.get("SSIM","")          if avg else "",
                    "LPIPS":          avg.get("LPIPS","")         if avg else "",
                    "CLIP_Score":     avg.get("CLIP_Score","")    if avg else "",
                    "weighted_score": avg.get("weighted_score","")if avg else "",
                    "n": n, "skipped": skipped}

        if avg_rm:  csv_rm.append(_row(avg_rm,  "object_removal",     len(rm_recs)))
        if avg_rp:  csv_rp.append(_row(avg_rp,  "object_replacement", len(rp_recs)))
        if avg_all: csv_all.append(_row(avg_all, "all",                n_ok))

    # ── 打印分组分任务表格 ─────────────────────────────────────────────────
    for group, gdata in summary.items():
        rm_rows = [(eid, d["name"], d["avg_removal"],     d["n_success"], d["skipped"])
                   for eid, d in gdata.items()]
        rp_rows = [(eid, d["name"], d["avg_replacement"], d["n_success"], d["skipped"])
                   for eid, d in gdata.items()]
        print_task_table(
            f"[{group}] object_removal  "
            f"| 权重: LPIPS×0.40  CLIP×0.40  PSNR×0.10  SSIM×0.10", rm_rows)
        print_task_table(
            f"[{group}] object_replacement  "
            f"| 权重: 各×0.25 等权", rp_rows)

    # ── 最优配置推荐 ────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  最优模块组合推荐（按 weighted_score 降序）")
    print(f"{'='*80}")

    for task_t, avg_key in (("object_removal",     "avg_removal"),
                             ("object_replacement", "avg_replacement")):
        best_score, best_eid, best_edata = -1, None, None
        for gdata in summary.values():
            for eid, edata in gdata.items():
                if edata["skipped"]: continue
                avg = edata.get(avg_key)
                if avg and avg["weighted_score"] > best_score:
                    best_score = avg["weighted_score"]
                    best_eid   = eid
                    best_edata = edata

        wkey = TASK_WEIGHT_KEY.get(task_t, "replace")
        w    = TASK_WEIGHTS[wkey]
        print(f"\n  [{task_t}]")
        print(f"  权重: PSNR×{w['PSNR']} SSIM×{w['SSIM']} "
              f"LPIPS×{w['LPIPS']} CLIP_Score×{w['CLIP_Score']}")
        if best_eid:
            p = merge_params(ABLATION_CONFIGS[best_eid])
            flags = [("M1-DSP_Mask",    p["use_dsp_mask"]),
                     ("M2-ControlNet",  p["use_controlnet"]),
                     ("M3-DSP_Enhance", p["use_dsp_enhance"]),
                     ("M4-Lap_Blend",   p["use_lap_blend"]),
                     ("M5-ConvexHull",  p["use_convex_hull"]),
                     ("M6-TwoStage",    p["use_two_stage"]),
                     ("M7-RmPrompt",    p["use_removal_prompt"]),
                     ("M8-CNShield",    p["controlnet_mask_shield"]),
                     ("M9-ColorMatch",  p["use_color_match"]),
                     ("M10-LaMa",       p["use_lama"]),
                     ("M11-OCR",        p["use_ocr_preprocess"])]
            on_m  = [n for n, v in flags if v]
            off_m = [n for n, v in flags if not v]
            print(f"  最优实验: {best_eid}  ({best_edata['name']})")
            print(f"  W.Score:  {best_score:.2f}")
            print(f"  开启: {', '.join(on_m) if on_m else '无（Baseline）'}")
            print(f"  关闭: {', '.join(off_m)}")
        else:
            print("  无有效结果（可能全部被跳过或无对应任务类型样本）")

    # ── 保存报告 ─────────────────────────────────────────────────────────────
    rpt = os.path.join(out_base, "ablation_report.json")
    with open(rpt, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": ts, "run_groups": list(run_groups),
            "task_filter": task_filter, "task_weights": TASK_WEIGHTS,
            "n_experiments": len(sel), "n_test_cases": len(all_tests),
            "summary": summary,
        }, f, indent=2, ensure_ascii=False, default=_safe)
    print(f"\n  ✓ 详细报告: {rpt}")

    save_csv(csv_rm,  os.path.join(out_base, "ablation_summary_removal.csv"))
    save_csv(csv_rp,  os.path.join(out_base, "ablation_summary_replacement.csv"))
    save_csv(csv_all, os.path.join(out_base, "ablation_summary_all.csv"))

    print(f"  ✓ 输出目录: {out_base}")
    print(f"{'='*80}\n")
    return summary


# =============================================================================
# 命令行入口
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Grounded-SAM Inpainting 消融实验")
    p.add_argument("--run_group", nargs="+",
                   default=["incremental", "leave_one_out", "key_combo"],
                   choices=["incremental","leave_one_out","key_combo","controlnet","all"])
    p.add_argument("--task_filter", default="all",
                   choices=["all","object_removal","object_replacement"])
    p.add_argument("--configs_subset", nargs="*", default=None,
                   help="仅运行指定实验 ID，如: A00_baseline C00_full")
    p.add_argument("--no_metrics",  action="store_true", help="跳过指标计算（调试用）")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--image_root",  default=None)
    p.add_argument("--prompts_config", default=None)
    p.add_argument("--sd15_model_path", default=None)
    p.add_argument("--output_dir",  default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for k in ("image_root", "prompts_config", "sd15_model_path"):
        v = getattr(args, k, None)
        if v: PATH_CONFIG[k] = v
    if args.output_dir:
        PATH_CONFIG["output_base_dir"] = args.output_dir

    groups = (list({ABLATION_CONFIGS[e]["group"] for e in ABLATION_CONFIGS})
              if "all" in args.run_group else args.run_group)

    run_ablation(
        run_groups     = groups,
        task_filter    = args.task_filter,
        configs_subset = args.configs_subset,
        enable_metrics = not args.no_metrics,
        device         = args.device,
    )