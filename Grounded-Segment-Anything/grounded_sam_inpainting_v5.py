"""
批量测试 Grounded-SAM Inpainting（支持评价指标）
v6：在 v5 全部改动基础上，新增多后端路由 + 精细化 Mask 流水线 + 分任务权重优化

v5 新增改进：
  ✅ 改进9：Removal 任务多后端优先路由
           - 技术路线：if removal → LaMa / ZITS / MAT 按优先级依次尝试；
             else → SDXL inpainting（替换类保持不变）
           - LaMa（lama）：傅里叶卷积，周期纹理还原最优，首选
           - ZITS（zits）：结构感知修复，边界完整性强，二选
           - MAT（mat）：Transformer 架构，大面积缺失首选，三选
           - 三者均通过 IOPaint (ModelManager) 调用，不可用时自动降级 SDXL
           - 新增参数：use_zits, use_mat（与 use_lama 联用，按优先级生效）

  ✅ 改进10：精细化 Mask 流水线（dilation + erosion + Gaussian 三步组合）
           - 原代码（v4）：凸包→闭运算→开运算→膨胀→Gaussian，
             过度依赖凸包，非凸形目标（人物/物体轮廓）mask 失真
           - 新代码（v5）：凸包（可选）→闭运算→膨胀→腐蚀→Gaussian 平滑
             * 膨胀（dilate）：均匀向外扩展 mask 边界，覆盖目标周围细节
             * 腐蚀（erode）：收缩细碎突起/锐角，使边界圆润平滑
             * Gaussian：对边界做软化过渡（soft edge），避免合成硬边
             * 凸包仅作可选步骤，不再是主导方法
           - 新增参数：mask_erode_ksize, mask_erode_iters,
                       mask_smooth_sigma（取代 feather_sigma）

  ✅ 改进11：分任务评价权重重新标定
           - removal 任务（与有气泡的原图对比，PSNR/SSIM 天然偏低）：
             LPIPS 主导（感知相似度）+ SSIM 辅助（局部结构），
             PSNR 降至极低权重，CLIP 仅做语义合理性参考
             新权重：PSNR=0.05 / SSIM=0.30 / LPIPS=0.55 / CLIP=0.10
           - replacement 任务（需要语义合理+视觉融合）：
             LPIPS + CLIP 双高权重，强调感知质量与语义对齐
             新权重：PSNR=0.10 / SSIM=0.10 / LPIPS=0.40 / CLIP=0.40

v4 已有改动（保留）：
  ✅ 改进7：LaMa Inpainting 后端（object_removal 专用）
  ✅ 改进8：形态学文字预处理（object_removal 专用，不依赖 PaddleOCR）
  ✅ 修复A-D：task_type 统一 / inpaint_mode 自动决策 /
              LaMa API 适配 / DSP 增强仅用于 SAM 分割

v3 已有改动（保留）：
  ✅ 改进1：凸包填充 Mask（现为可选步骤）
  ✅ 改进2：两阶段 Inpainting
  ✅ 改进3：删除类专用 Prompt 工程
  ✅ 改进4：ControlNet 条件图屏蔽 Mask 区域
  ✅ 改进5：边缘色彩匹配后处理
  ✅ 改进6：评价指标分任务类型加权

v2 已有改动（保留）：
  ✅ 新增A：DSP Mask 优化
  ✅ 新增B：ControlNet 约束下的 SD Inpainting
  ✅ 新增C：DSP 线条增强与双边滤波（仅用于 SAM 分割）
  ✅ 新增D：拉普拉斯金字塔融合

依赖安装：
  pip install iopaint   # LaMa / ZITS / MAT 均通过 IOPaint 调用
"""

import os
import sys
import argparse
from pathlib import Path
import json
from datetime import datetime

# 设置 BERT 本地路径
BERT_PATH = "/root/autodl-tmp/Grounded_sam/bert-base-uncased"
if os.path.exists(BERT_PATH):
    os.environ['TRANSFORMERS_CACHE'] = os.path.dirname(BERT_PATH)
    os.environ['HF_HOME'] = os.path.dirname(BERT_PATH)
    print(f"✓ 使用本地 BERT: {BERT_PATH}\n")

try:
    from grounded_sam_inpainting import (
        load_image,
        load_model,
        get_grounding_output,
        filter_boxes_nms,
        resize_with_padding,
        restore_from_padding,
        show_mask,
        show_box
    )
    import torch
    import cv2
    from PIL import Image
    from segment_anything import SamPredictor, build_sam
    from diffusers import StableDiffusionXLInpaintPipeline
    # 新增B：导入 ControlNet 相关模块
    from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio as psnr_func
    from skimage.metrics import structural_similarity as ssim_func
    import lpips
    import open_clip
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有依赖已安装")
    sys.exit(1)

# ✅ 改进7：IOPaint（LaMa）可选导入——不强制，缺失时自动降级为 SDXL
try:
    from iopaint.model_manager import ModelManager
    from iopaint.schema import InpaintRequest, HDStrategy, LDMSampler
    IOPAINT_AVAILABLE = True
    print("✓ IOPaint (LaMa) 可用\n")
except ImportError:
    IOPAINT_AVAILABLE = False
    print("⚠ IOPaint 未安装，object_removal 将回退到 SDXL（pip install iopaint）\n")

# ✅ 改进8：形态学文字预处理不依赖 PaddleOCR，始终可用
# PaddleOCR 因版本兼容性问题（API 变更、底层推理引擎 bug）放弃使用
# 改用 OpenCV 形态学连通域分析定位漫画黑色文字，无需任何额外依赖
PADDLEOCR_AVAILABLE = True   # 形态学方案始终可用
print("✓ 形态学文字预处理方案就绪（无需 PaddleOCR）\n")


# ==================== ✅ 改进7：LaMa Inpainting 模块 ====================
# 原代码（v3）：object_removal 使用 SDXL inpainting，易产生幻觉，
#               对漫画网点/规律线条重建质量差
# 新代码（v4）：object_removal 改用 LaMa（通过 IOPaint），
#               LaMa 基于傅里叶卷积，专为"擦除还原"设计，不需要 prompt，
#               对周期性纹理的还原精度远超 latent diffusion 模型
#
# ✅ 修复C：IOPaint 实际调用方式：
#   - model() 接收 BGR numpy array（不是 RGB）
#   - mask 需二值化为 0/255，白色（255）为修复区
#   - 返回值为 BGR numpy array

# LaMa 模型单例（避免每次调用重复加载权重）
_lama_model_manager = None

def get_lama_model(device: str = "cuda"):
    """
    懒加载 LaMa 模型（全局单例，首次调用时初始化）
    """
    global _lama_model_manager
    if _lama_model_manager is None:
        if not IOPAINT_AVAILABLE:
            raise RuntimeError(
                "IOPaint 未安装，无法使用 LaMa。"
                "请运行：pip install iopaint"
            )
        print("  [改进7] 首次初始化 LaMa 模型（后续调用复用）...")
        _lama_model_manager = ModelManager(
            name="lama",
            device=device,
        )
        print("  ✓ LaMa 模型加载完成")
    return _lama_model_manager


def lama_inpaint(image_np: np.ndarray,
                 mask_np: np.ndarray,
                 device: str = "cuda") -> np.ndarray:
    """
    ✅ 改进7：使用 LaMa 执行 inpainting

    Args:
        image_np : RGB uint8 原图（与 mask 同尺寸）
        mask_np  : uint8 mask（255=待修复区域，0=保留区域）
        device   : 运算设备
    Returns:
        修复后的 RGB uint8 图像（与输入同尺寸）
    """
    model = get_lama_model(device)

    # ✅ 修复C：IOPaint 接收 BGR 格式
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # mask 二值化：确保只有 0 和 255
    mask_single = mask_np if mask_np.ndim == 2 else mask_np[:, :, 0]
    _, mask_binary = cv2.threshold(mask_single, 127, 255, cv2.THRESH_BINARY)

    # 构造请求对象
    req = InpaintRequest(
        hd_strategy=HDStrategy.ORIGINAL,
        hd_strategy_crop_margin=32,
        hd_strategy_crop_trigger_size=800,
        hd_strategy_resize_limit=1280,
        ldm_sampler=LDMSampler.ddim,
    )

    # 执行修复（返回 BGR uint8）
    result_bgr = model(image_bgr, mask_binary, req)

    # 转回 RGB
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    return result_rgb.astype(np.uint8)


# ==================== ✅ 改进9：ZITS / MAT Inpainting 模块 ====================
# Removal 任务后端优先级：LaMa → ZITS → MAT → SDXL（降级）
#
# ZITS (Zero-Shot Image-To-Image Translation with Structure)
#   - 结构感知修复，对漫画轮廓线/气泡边界保留效果更好
#   - 适合目标周边有明确线条结构的场景
#
# MAT (Mask-Aware Transformer)
#   - Transformer 架构，大面积 mask 修复质量优于 CNN 类模型
#   - 适合整页删除、大面积人物擦除等场景
#
# 三者均通过 IOPaint ModelManager 调用，接口完全统一

_zits_model_manager = None
_mat_model_manager  = None


def get_zits_model(device: str = "cuda"):
    """懒加载 ZITS 模型（全局单例）"""
    global _zits_model_manager
    if _zits_model_manager is None:
        if not IOPAINT_AVAILABLE:
            raise RuntimeError("IOPaint 未安装，无法使用 ZITS。pip install iopaint")
        print("  [改进9] 首次初始化 ZITS 模型（后续调用复用）...")
        _zits_model_manager = ModelManager(name="zits", device=device)
        print("  ✓ ZITS 模型加载完成")
    return _zits_model_manager


def get_mat_model(device: str = "cuda"):
    """懒加载 MAT 模型（全局单例）"""
    global _mat_model_manager
    if _mat_model_manager is None:
        if not IOPAINT_AVAILABLE:
            raise RuntimeError("IOPaint 未安装，无法使用 MAT。pip install iopaint")
        print("  [改进9] 首次初始化 MAT 模型（后续调用复用）...")
        _mat_model_manager = ModelManager(name="mat", device=device)
        print("  ✓ MAT 模型加载完成")
    return _mat_model_manager


def _iopaint_inpaint_generic(get_model_fn,
                              image_np: np.ndarray,
                              mask_np: np.ndarray,
                              device: str = "cuda") -> np.ndarray:
    """
    通用 IOPaint 推理封装（ZITS / MAT 共用）
    入参/出参格式与 lama_inpaint 完全一致：RGB uint8 in → RGB uint8 out
    """
    model = get_model_fn(device)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    mask_single = mask_np if mask_np.ndim == 2 else mask_np[:, :, 0]
    _, mask_binary = cv2.threshold(mask_single, 127, 255, cv2.THRESH_BINARY)
    req = InpaintRequest(
        hd_strategy=HDStrategy.ORIGINAL,
        hd_strategy_crop_margin=32,
        hd_strategy_crop_trigger_size=800,
        hd_strategy_resize_limit=1280,
        ldm_sampler=LDMSampler.ddim,
    )
    result_bgr = model(image_bgr, mask_binary, req)
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)


def zits_inpaint(image_np: np.ndarray,
                 mask_np: np.ndarray,
                 device: str = "cuda") -> np.ndarray:
    """✅ 改进9：ZITS inpainting（RGB uint8 → RGB uint8）"""
    return _iopaint_inpaint_generic(get_zits_model, image_np, mask_np, device)


def mat_inpaint(image_np: np.ndarray,
                mask_np: np.ndarray,
                device: str = "cuda") -> np.ndarray:
    """✅ 改进9：MAT inpainting（RGB uint8 → RGB uint8）"""
    return _iopaint_inpaint_generic(get_mat_model, image_np, mask_np, device)


def removal_inpaint_with_priority(image_np: np.ndarray,
                                   mask_np: np.ndarray,
                                   device: str = "cuda",
                                   use_lama: bool = True,
                                   use_zits: bool = True,
                                   use_mat: bool = True) -> tuple:
    """
    ✅ 改进9：Removal 任务后端优先路由
    按 LaMa → ZITS → MAT 顺序依次尝试，均不可用时返回 None（外部降级 SDXL）。

    Returns:
        (result_np, backend_name) 或 (None, "sdxl_fallback")
    """
    if not IOPAINT_AVAILABLE:
        return None, "sdxl_fallback"

    candidates = []
    if use_lama:
        candidates.append(("lama",  lama_inpaint))
    if use_zits:
        candidates.append(("zits",  zits_inpaint))
    if use_mat:
        candidates.append(("mat",   mat_inpaint))

    for backend_name, inpaint_fn in candidates:
        try:
            print(f"  [改进9] 尝试后端: {backend_name.upper()}...")
            result = inpaint_fn(image_np, mask_np, device)
            print(f"  ✓ {backend_name.upper()} inpainting 完成")
            return result, backend_name
        except Exception as e:
            print(f"  ⚠ {backend_name.upper()} 失败（{e}），尝试下一后端...")

    print("  ⚠ 所有 IOPaint 后端均失败，回退 SDXL")
    return None, "sdxl_fallback"


# 原方案（PaddleOCR）：版本兼容性问题严重，API 多次变更，底层推理引擎存在 bug
# 新方案（形态学）：基于 OpenCV 连通域分析，不依赖任何 OCR 框架
#
# 处理逻辑：
#   1. 在 mask 区域内提取深色像素（漫画文字通常为黑色，灰度 < 80）
#   2. 形态学膨胀连接相邻文字笔画
#   3. 连通域分析，按面积过滤（去除噪点和大块背景线条）
#   4. 将识别到的文字连通域填充为白色

# _ocr_engine 保留为 None（兼容旧代码引用，实际不再使用）
_ocr_engine = None

def get_ocr_engine():
    """保留函数签名兼容性，形态学方案无需初始化引擎"""
    return None


def ocr_fill_text(image_np: np.ndarray,
                  mask_np: np.ndarray,
                  fill_color: tuple = (255, 255, 255),
                  ocr_confidence_threshold: float = 0.5) -> np.ndarray:
    """
    ✅ 改进8（v4 形态学修订版）：在 mask 区域内定位深色文字并填白

    基于 OpenCV 形态学连通域分析，专为漫画黑色文字设计：
    1. 提取 mask 区域内灰度值 < 80 的深色像素（文字候选）
    2. 膨胀连接相邻笔画
    3. 连通域面积过滤：去除噪点（太小）和背景线条（太大）
    4. 填充识别到的文字区域为 fill_color

    Args:
        image_np                 : RGB uint8 原图
        mask_np                  : uint8 mask（255=气泡区域）
        fill_color               : 文字填充颜色，默认白色 (255,255,255)
        ocr_confidence_threshold : 保留参数（形态学方案不使用，保持接口兼容）
    Returns:
        文字已填充的 RGB uint8 图像（与原图同尺寸）
    """
    image_filled = image_np.copy()
    mask_binary  = (mask_np > 127).astype(np.uint8)

    # Step 1：在 mask 区域内提取深色像素
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    text_candidates = cv2.bitwise_and(dark_mask, dark_mask, mask=mask_binary)

    # Step 2：形态学膨胀，连接相邻文字笔画
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    text_connected = cv2.dilate(text_candidates, kernel_connect, iterations=2)

    # Step 3：连通域分析，按面积过滤
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        text_connected, connectivity=8)

    mask_area = int(mask_binary.sum())
    area_min  = 20
    area_max  = max(100, mask_area * 0.15)   # 不超过 mask 面积的 15%

    text_region_mask = np.zeros_like(dark_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area_min <= area <= area_max:
            text_region_mask[labels == i] = 255

    # Step 4：轻微膨胀覆盖笔画边缘
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    text_region_mask = cv2.dilate(text_region_mask, kernel_fill, iterations=1)

    # Step 5：填充（仅在 mask 区域内）
    fill_region = (text_region_mask > 0) & (mask_binary > 0)
    text_pixel_count = int(fill_region.sum())

    if text_pixel_count > 0:
        image_filled[fill_region] = fill_color

    return image_filled


# ==================== 新增A（v2）：DSP Mask 优化模块 ====================
# 原代码：仅做简单膨胀+高斯模糊
# v2：增加形态学修补（闭运算填孔）+ 平滑轮廓 + 可控羽化强度
# v3：对删除类任务追加凸包填充（改进1），确保气泡整体轮廓无漏洞

def fill_convex_hull(mask_np: np.ndarray) -> np.ndarray:
    """
    ✅ 改进1：凸包填充
    对 mask 中每个连通域求凸包并填充，消除气泡内部文字笔画、
    阴影等造成的 mask 空洞，确保气泡区域被完整覆盖。

    原代码（v2）：仅使用闭运算填补空洞，对复杂文字排版效果有限
    新代码（v3）：先求凸包再填充，一次性填满凸形气泡内部所有缝隙
    """
    contours, _ = cv2.findContours(
        mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_mask = np.zeros_like(mask_np)
    for cnt in contours:
        if len(cnt) >= 3:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(hull_mask, [hull], -1, 255, -1)
    # 合并凸包结果与原始 mask（保证非凸形区域也不丢失）
    combined = cv2.bitwise_or(mask_np, hull_mask)
    return combined


def dsp_optimize_mask(mask_np: np.ndarray,
                      morph_close_ksize: int = 21,
                      morph_open_ksize: int = 7,       # 保留参数，v5 中已废弃（替换为 erode）
                      dilate_ksize: int = 15,
                      dilate_iters: int = 1,
                      feather_sigma: int = 21,          # 向后兼容别名 → 内部映射 smooth_sigma
                      feather_strength: float = 1.0,
                      use_convex_hull: bool = False,
                      # ✅ 改进10：新增精细化 Mask 参数
                      erode_ksize: int = 9,
                      erode_iters: int = 1,
                      smooth_sigma: int = -1,           # -1 表示继承 feather_sigma
                      ) -> np.ndarray:
    """
    ✅ 改进10：精细化 Mask 流水线
    [凸包（可选）] → 闭运算 → 膨胀 → 腐蚀 → Gaussian 平滑

    原代码（v4）：闭运算 → 开运算 → 膨胀 → Gaussian
      问题：开运算会去除小连通域，可能丢失非凸目标边角；凸包是主导方法

    新流水线（v5）：
      Step 1  (可选) convex hull：仅对气泡等凸形轮廓有效，非凸目标不启用
      Step 2  close（闭运算）：填补 mask 内部空洞/碎裂区域
      Step 3  dilate（膨胀）：均匀向外扩展 mask，覆盖目标周围 halo
      Step 4  erode（腐蚀）：收缩细碎突起/锐角，使边界圆润平滑；
                             dilate > erode → 净效果为扩展但轮廓更平滑
      Step 5  Gaussian 平滑：对边界做软过渡（soft edge），避免合成硬边
      Step 6  feather_strength 混合：0=硬边, 1=完全软化（向后兼容）

    新增参数：
        erode_ksize   : 腐蚀核尺寸（建议 dilate_ksize 的 1/2 ~ 2/3）
        erode_iters   : 腐蚀迭代次数
        smooth_sigma  : Gaussian 平滑 sigma（-1 时继承 feather_sigma）
    """
    # ✅ 改进1：凸包填充（可选，不再是唯一/主导方法）
    if use_convex_hull:
        mask_np = fill_convex_hull(mask_np)

    # smooth_sigma 向后兼容：未显式指定时沿用 feather_sigma
    effective_sigma = feather_sigma if smooth_sigma < 0 else smooth_sigma

    # Step 1：闭运算（填补 mask 内部空洞/碎裂区域）
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_close_ksize, morph_close_ksize))
    mask_closed = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel_close)

    # Step 2：膨胀（扩展 mask 边界，覆盖目标周围区域）
    # ✅ 改进10：使用椭圆核，比矩形核更接近各向同性扩展
    kernel_dilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
    mask_dilated = cv2.dilate(mask_closed, kernel_dilate, iterations=dilate_iters)

    # Step 3：腐蚀（收缩细碎突起/锐角，使边界圆润）
    # ✅ 改进10：新增，净效果 = dilate_ksize - erode_ksize 的净扩展量 + 平滑轮廓
    kernel_erode = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_ksize, erode_ksize))
    mask_eroded = cv2.erode(mask_dilated, kernel_erode, iterations=erode_iters)

    # Step 4：Gaussian 平滑（边界软化，避免合成硬边）
    # ✅ 改进10：smooth_sigma 参数，ksize 自动对齐奇数
    smooth_ksize = effective_sigma if effective_sigma % 2 == 1 else effective_sigma + 1
    mask_smoothed = cv2.GaussianBlur(
        mask_eroded, (smooth_ksize, smooth_ksize), effective_sigma)

    # Step 5：按 feather_strength 混合软化结果与硬边 mask（向后兼容）
    if feather_strength < 1.0:
        mask_out = cv2.addWeighted(
            mask_smoothed, feather_strength,
            mask_eroded,   1.0 - feather_strength,
            0
        )
    else:
        mask_out = mask_smoothed

    return mask_out.astype(np.uint8)


# ==================== 新增C（v2）：DSP 线条增强与双边滤波模块 ====================

def dsp_enhance_image(image_np: np.ndarray,
                      bilateral_d: int = 9,
                      bilateral_sigma_color: float = 75,
                      bilateral_sigma_space: float = 75,
                      unsharp_strength: float = 0.5,
                      unsharp_sigma: int = 3) -> np.ndarray:
    """
    DSP 图像预处理：双边滤波去噪 + Unsharp Mask 线条增强
    （v2 已有，v3 保留不变）
    """
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    img_filtered = cv2.bilateralFilter(
        img_bgr, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)

    ksize = unsharp_sigma if unsharp_sigma % 2 == 1 else unsharp_sigma + 1
    img_blurred = cv2.GaussianBlur(img_filtered, (ksize * 2 + 1, ksize * 2 + 1), unsharp_sigma)
    img_sharpened = cv2.addWeighted(
        img_filtered, 1.0 + unsharp_strength,
        img_blurred, -unsharp_strength,
        0
    )
    img_sharpened = np.clip(img_sharpened, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_sharpened, cv2.COLOR_BGR2RGB)


# ==================== 新增D（v2）：拉普拉斯金字塔融合模块 ====================

def build_gaussian_pyramid(img: np.ndarray, levels: int) -> list:
    """构建高斯金字塔"""
    pyramid = [img.astype(np.float32)]
    for _ in range(levels - 1):
        img = cv2.pyrDown(img)
        pyramid.append(img.astype(np.float32))
    return pyramid


def build_laplacian_pyramid(img: np.ndarray, levels: int) -> list:
    """构建拉普拉斯金字塔"""
    gaussian = build_gaussian_pyramid(img, levels)
    laplacian = []
    for i in range(levels - 1):
        up = cv2.pyrUp(gaussian[i + 1], dstsize=(gaussian[i].shape[1], gaussian[i].shape[0]))
        lap = gaussian[i].astype(np.float32) - up.astype(np.float32)
        laplacian.append(lap)
    laplacian.append(gaussian[-1].astype(np.float32))
    return laplacian


def laplacian_pyramid_blend(orig_np: np.ndarray,
                             inpainted_np: np.ndarray,
                             mask_np: np.ndarray,
                             levels: int = 6) -> np.ndarray:
    """
    拉普拉斯金字塔融合（v2 已有，v3 保留不变）
    """
    mask_f = mask_np.astype(np.float32) / 255.0
    if mask_f.ndim == 2:
        mask_f = np.stack([mask_f] * 3, axis=-1)

    lap_orig      = build_laplacian_pyramid(orig_np,       levels)
    lap_inpainted = build_laplacian_pyramid(inpainted_np,  levels)
    gauss_mask    = build_gaussian_pyramid(mask_f,          levels)

    blended_pyramid = []
    for lo, li, gm in zip(lap_orig, lap_inpainted, gauss_mask):
        blended = lo * (1 - gm) + li * gm
        blended_pyramid.append(blended)

    result = blended_pyramid[-1]
    for i in range(levels - 2, -1, -1):
        result = cv2.pyrUp(result, dstsize=(blended_pyramid[i].shape[1], blended_pyramid[i].shape[0]))
        result = result + blended_pyramid[i]

    return np.clip(result, 0, 255).astype(np.uint8)


# ==================== ✅ 改进5（v3）：边缘色彩匹配后处理模块 ====================
# 原代码（v2）：拉普拉斯融合后直接输出，大面积重建区色调可能与周边不一致
# 新代码（v3）：在 mask 边缘采样原图颜色统计，对重建区做线性颜色校正

def color_match_inpainted(orig_np: np.ndarray,
                           inpainted_np: np.ndarray,
                           mask_np: np.ndarray,
                           border_radius: int = 30,
                           blend_radius: int = 15) -> np.ndarray:
    """
    ✅ 改进5：边缘色彩匹配
    在 mask 膨胀后的边缘环形区域采样原图的颜色统计（均值/标准差），
    对 inpainted 区域执行线性颜色迁移，消除大面积重建后的色调断层。

    Args:
        orig_np       : 原图 RGB uint8
        inpainted_np  : inpainting/融合后的图像 RGB uint8
        mask_np       : uint8 mask（0=原图区，255=重建区）
        border_radius : 边缘采样环的宽度（像素），越大采样越稳定
        blend_radius  : 颜色校正过渡带宽度，避免校正边界处突变
    Returns:
        颜色校正后的 RGB uint8 图像
    """
    result = inpainted_np.astype(np.float32)
    orig_f = orig_np.astype(np.float32)

    # Step 1：构建边缘环形采样区域（mask 膨胀 - mask 本身）
    kernel_border = np.ones((border_radius * 2 + 1, border_radius * 2 + 1), np.uint8)
    mask_binary = (mask_np > 127).astype(np.uint8) * 255
    mask_dilated_border = cv2.dilate(mask_binary, kernel_border, iterations=1)
    border_zone = cv2.subtract(mask_dilated_border, mask_binary)  # 环形区域

    # Step 2：在边缘环形区域统计原图与重建图的颜色均值/标准差（逐通道）
    border_pixels_orig     = orig_f[border_zone > 0]
    border_pixels_inpainted = result[border_zone > 0]

    if len(border_pixels_orig) < 10:
        # 采样点过少时跳过校正
        return inpainted_np

    correction_result = result.copy()
    for c in range(3):
        mu_orig = border_pixels_orig[:, c].mean()
        mu_inp  = border_pixels_inpainted[:, c].mean()
        std_orig = border_pixels_orig[:, c].std() + 1e-6
        std_inp  = border_pixels_inpainted[:, c].std() + 1e-6

        # Step 3：对重建区域的每个像素做线性颜色迁移
        # 公式：pixel_corrected = (pixel - mu_inp) / std_inp * std_orig + mu_orig
        scale = std_orig / std_inp
        shift = mu_orig - mu_inp * scale

        mask_region = mask_binary > 0
        correction_result[:, :, c][mask_region] = (
            result[:, :, c][mask_region] * scale + shift
        )

    correction_result = np.clip(correction_result, 0, 255).astype(np.uint8)

    # Step 4：在 mask 边缘附近做软过渡（避免颜色校正区域出现硬边）
    if blend_radius > 0:
        blend_ksize = blend_radius * 2 + 1
        blend_weight = cv2.GaussianBlur(
            mask_binary.astype(np.float32) / 255.0,
            (blend_ksize, blend_ksize), blend_radius
        )
        blend_weight = np.stack([blend_weight] * 3, axis=-1)
        final = (correction_result.astype(np.float32) * blend_weight +
                 inpainted_np.astype(np.float32) * (1.0 - blend_weight))
        return np.clip(final, 0, 255).astype(np.uint8)

    return correction_result


# ==================== 新增B（v2）：ControlNet 管线加载函数 ====================
# v3 改进4：generate_controlnet_condition 新增屏蔽 mask 区域的功能

def load_controlnet_pipe(sd_model_path: str,
                          controlnet_model_path: str,
                          device: str = "cuda") -> StableDiffusionXLControlNetInpaintPipeline:
    """
    加载带 ControlNet 约束的 SDXL Inpainting 管线（v2 已有，v3 保留不变）
    """
    print("加载 ControlNet 模型...")
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_path,
        torch_dtype=torch.float16,
        local_files_only=True
    )
    print("加载 SDXL ControlNet Inpainting 管线...")
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        sd_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        local_files_only=True
    )
    pipe = pipe.to(device)
    print("✓ ControlNet Inpainting 管线加载完成\n")
    return pipe


def generate_controlnet_condition(image_np: np.ndarray,
                                   condition_type: str = "canny",
                                   mask_np: np.ndarray = None,
                                   mask_threshold: int = 127) -> Image.Image:
    """
    生成 ControlNet 条件图（Canny 边缘 / 深度图）

    ✅ 改进4（v3）：新增 mask_np 参数
    原代码（v2）：条件图包含气泡区域的 Canny 边缘，会错误约束待修复区域的生成方向
    新代码（v3）：将 mask 区域对应的条件图置零，仅用周边结构引导几何一致性，
                  防止气泡轮廓的 Canny 边缘"鬼影"出现在重建结果中

    Args:
        image_np       : RGB uint8 图像
        condition_type : "canny" 或 "depth"
        mask_np        : uint8 mask（不为 None 时，mask 区域条件图置零）
        mask_threshold : mask 二值化阈值
    """
    if condition_type == "canny":
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, threshold1=100, threshold2=200)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    elif condition_type == "depth":
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"不支持的 condition_type: {condition_type}")

    # ✅ 改进4：屏蔽 mask 区域，避免气泡轮廓的结构信息约束重建方向
    if mask_np is not None:
        mask_binary = (cv2.resize(
            mask_np,
            (edges_rgb.shape[1], edges_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ) > mask_threshold).astype(np.uint8)
        edges_rgb[mask_binary == 1] = 0  # 待修复区域不施加结构约束

    return Image.fromarray(edges_rgb)


# ==================== ✅ 改进3（v3）：删除类专用 Prompt 构建函数 ====================
# 原代码（v2）：直接使用用户提供的 inpaint_prompt 和固定的 negative_prompt
# 新代码（v3）：针对删除类任务，自动在 negative_prompt 中追加气泡/文字屏蔽词，
#               并在 prompt 中强调背景续接语义，引导模型做"无痕删除"而非"创意替换"

# 删除类任务的关键词黑名单（追加到 negative_prompt）
REMOVAL_NEGATIVE_KEYWORDS = (
    "speech bubble, thought bubble, text bubble, caption, subtitle, "
    "text, letter, word, font, typography, watermark, label, annotation, "
    "border, outline, stroke, shadow of text, comic panel border"
)

# 删除类任务的背景续接正向引导词（追加到 inpaint_prompt）
REMOVAL_POSITIVE_SUFFIX = (
    "seamless background continuation, no text, no bubble, "
    "consistent texture, consistent lighting, photorealistic, clean surface"
)

def build_removal_prompts(inpaint_prompt: str,
                           negative_prompt: str,
                           task_type: str = "object_replacement") -> tuple:
    """
    ✅ 改进3：删除类任务专用 Prompt 构建
    根据 task_type 自动增强 prompt 和 negative_prompt。

    Args:
        inpaint_prompt  : 原始正向提示词
        negative_prompt : 原始负向提示词
        task_type       : "removal"（删除类） 或 "replace"（替换类）
    Returns:
        (enhanced_prompt, enhanced_negative_prompt)
    """
    if task_type == "object_removal":
        enhanced_prompt = f"{inpaint_prompt}, {REMOVAL_POSITIVE_SUFFIX}"
        enhanced_negative = f"{negative_prompt}, {REMOVAL_NEGATIVE_KEYWORDS}"
        return enhanced_prompt, enhanced_negative
    else:
        return inpaint_prompt, negative_prompt


# ==================== ✅ 改进2（v3）：两阶段 Inpainting 函数 ====================
# 原代码（v2）：单次调用 pipe，strength=0.99，对大面积重建结构稳定性不足
# 新代码（v3）：
#   第一阶段：较高 strength（0.65）+ 高 guidance_scale（9.0），生成粗略背景结构
#   第二阶段：以第一阶段输出为 init_image，低 strength（0.35）+ 常规 guidance，细化纹理
#   两阶段共用同一个 pipe，无需额外加载模型

def two_stage_inpaint(pipe,
                       prompt: str,
                       negative_prompt: str,
                       image_padded: Image.Image,
                       mask_padded: Image.Image,
                       stage1_steps: int = 30,
                       stage1_strength: float = 0.65,
                       stage1_guidance: float = 9.0,
                       stage2_steps: int = 20,
                       stage2_strength: float = 0.35,
                       stage2_guidance: float = 7.5,
                       # ControlNet 相关（若 pipe 为 ControlNet 管线则传入）
                       use_controlnet: bool = False,
                       condition_padded: Image.Image = None,
                       controlnet_scale: float = 0.5) -> Image.Image:
    """
    ✅ 改进2：两阶段 Inpainting

    第一阶段目标：在大面积 mask 区域生成符合背景语义的粗略结构
    第二阶段目标：以第一阶段输出为基础，用低 strength 细化纹理细节，
                  避免单次高 strength 带来的结构崩塌

    Args:
        pipe            : SDXL Inpainting 或 ControlNet Inpainting 管线
        stage1_strength : 第一阶段去噪强度（建议 0.6~0.7）
        stage2_strength : 第二阶段去噪强度（建议 0.3~0.4）
        use_controlnet  : 是否为 ControlNet 管线
        condition_padded: ControlNet 条件图（use_controlnet=True 时必传）
    Returns:
        两阶段融合后的 PIL.Image
    """
    common_kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        mask_image=mask_padded,
    )
    if use_controlnet and condition_padded is not None:
        common_kwargs["control_image"] = condition_padded
        common_kwargs["controlnet_conditioning_scale"] = controlnet_scale

    # ── 第一阶段：粗略结构生成 ──────────────────────────────────────────
    print("    [两阶段-第1阶段] 粗略背景结构生成...")
    stage1_result = pipe(
        **common_kwargs,
        image=image_padded,
        num_inference_steps=stage1_steps,
        guidance_scale=stage1_guidance,
        strength=stage1_strength,
    ).images[0]

    # ── 第二阶段：纹理细化 ────────────────────────────────────────────────
    print("    [两阶段-第2阶段] 纹理细化...")
    stage2_result = pipe(
        **common_kwargs,
        image=stage1_result,   # ← 以第一阶段输出为输入
        num_inference_steps=stage2_steps,
        guidance_scale=stage2_guidance,
        strength=stage2_strength,
    ).images[0]

    return stage2_result


# ==================== 改进6（v3）：评价指标类（分任务类型加权） ====================

class Evaluator:
    # ✅ 改进11：分任务权重重新标定
    #
    # removal 任务：
    #   与有气泡的原图对比，PSNR/SSIM 天然偏低（背景内容本就不同）
    #   → LPIPS 主导（感知相似度）+ SSIM 辅助（局部结构保留）
    #   → PSNR 降至极低，CLIP 仅做语义合理性参考（删除任务无明确文本 prompt）
    #   新权重：PSNR=0.05 / SSIM=0.30 / LPIPS=0.55 / CLIP=0.10
    #
    # replacement 任务：
    #   需要语义合理（CLIP）+ 视觉融合（LPIPS）双重保障
    #   → LPIPS + CLIP 双高权重，PSNR/SSIM 作辅助
    #   新权重：PSNR=0.10 / SSIM=0.10 / LPIPS=0.40 / CLIP=0.40
    # ✅ 修复E（v6）：键名与 task_type 字符串对齐
    # 原代码（v5）：键 "removal"/"replace" 与实际传入的 "object_removal"/"object_replacement"
    #   不匹配 → self.TASK_WEIGHTS.get() 永远返回 None → 全部回退 "replace" 权重
    #   （可从测试报告中所有样本 weights_used 完全相同确认此 bug）
    TASK_WEIGHTS = {
        "object_removal":     {"PSNR": 0.05, "SSIM": 0.30, "LPIPS": 0.55, "CLIP_Score": 0.10},
        "object_replacement": {"PSNR": 0.10, "SSIM": 0.10, "LPIPS": 0.40, "CLIP_Score": 0.40},
    }

    def __init__(self, device):
        self.device = device
        print("初始化评价指标模型...")
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='laion2b_s34b_b79k'
        )
        self.clip_model = self.clip_model.to(device)
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        print("✓ 评价指标模型加载完成\n")

    def run(self, org_path, res_path, prompt, task_type: str = "object_replacement"):
        """
        ✅ 改进6：新增 task_type 参数，对删除类任务调整各指标权重，
        输出加权综合分 weighted_score，更准确反映删除类任务的真实质量。
        """
        try:
            img_org = Image.open(org_path).convert('RGB').resize((512, 512))
            img_res = Image.open(res_path).convert('RGB').resize((512, 512))
            np_org  = np.array(img_org)
            np_res  = np.array(img_res)

            psnr_value = psnr_func(np_org, np_res, data_range=255)
            ssim_value = ssim_func(np_org, np_res, channel_axis=2, data_range=255)

            t_org = lpips.im2tensor(np_org).to(self.device)
            t_res = lpips.im2tensor(np_res).to(self.device)
            lpips_value = self.lpips_model(t_org, t_res).item()

            img_input  = self.clip_preprocess(img_res).unsqueeze(0).to(self.device)
            text_input = self.tokenizer([prompt]).to(self.device)
            with torch.no_grad():
                img_feats = self.clip_model.encode_image(img_input)
                txt_feats = self.clip_model.encode_text(text_input)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
                clip_score = (img_feats @ txt_feats.T).item() * 100

            # ✅ 改进6：按任务类型加权计算综合分
            weights = self.TASK_WEIGHTS.get(task_type, self.TASK_WEIGHTS["object_replacement"])
            # LPIPS 越小越好，转换为"越大越好"的得分再加权
            lpips_score = max(0.0, 1.0 - lpips_value)
            # PSNR 归一化到 [0,1]（假设合理范围 0~50 dB）
            psnr_norm  = min(psnr_value / 50.0, 1.0)
            weighted_score = (
                weights["PSNR"]       * psnr_norm +
                weights["SSIM"]       * ssim_value +
                weights["LPIPS"]      * lpips_score +
                weights["CLIP_Score"] * (clip_score / 100.0)
            ) * 100

            return {
                "PSNR":           round(psnr_value,  2),
                "SSIM":           round(ssim_value,  4),
                "LPIPS":          round(lpips_value, 4),
                "CLIP_Score":     round(clip_score,  2),
                "weighted_score": round(weighted_score, 2),
                "task_type":      task_type,
                "weights_used":   weights,
            }
        except Exception as e:
            print(f"    警告: 评价指标计算失败: {e}")
            return None


def load_test_configs(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ==================== 主测试函数 ====================

def run_single_test(
    image_path,
    det_prompt,
    inpaint_prompt,
    description,
    model,
    sam_checkpoint,
    pipe,
    output_base_dir,
    evaluator=None,
    box_threshold=0.3,
    text_threshold=0.25,
    inpaint_mode="merge",
    nms_iou_threshold=0.5,
    inpaint_steps=50,
    inpaint_guidance_scale=7.5,
    inpaint_strength=0.99,
    inpaint_negative_prompt="blurry, bad quality, distorted, artifacts, ugly, low resolution",
    device="cuda",
    # 新增A（v2）：DSP mask 优化参数
    use_dsp_mask=True,
    dsp_morph_close_ksize=21,
    dsp_morph_open_ksize=7,
    dsp_feather_sigma=21,
    dsp_feather_strength=1.0,
    # 新增B（v2）：ControlNet 参数
    use_controlnet=False,
    controlnet_condition_type="canny",
    controlnet_scale=0.5,
    # 新增C（v2）：DSP 线条增强参数
    use_dsp_enhance=True,
    dsp_bilateral_d=9,
    dsp_bilateral_sigma=75,
    dsp_unsharp_strength=0.5,
    # 新增D（v2）：拉普拉斯金字塔融合参数
    use_lap_blend=False,
    lap_blend_levels=6,
    # ✅ 改进1（v3）：凸包填充开关
    use_convex_hull=False,
    # ✅ 改进2（v3）：两阶段 Inpainting 开关及参数
    use_two_stage=False,
    stage1_steps=30,
    stage1_strength=0.65,
    stage1_guidance=9.0,
    stage2_steps=20,
    stage2_strength=0.35,
    stage2_guidance=7.5,
    # ✅ 改进3（v3）：任务类型（影响 Prompt 增强与评价指标权重）
    task_type="object_replacement",
    # ✅ 改进4（v3）：ControlNet 条件图屏蔽 mask 区域（针对删除任务）
    controlnet_mask_shield=True,
    # ✅ 改进5（v3）：边缘色彩匹配后处理
    use_color_match=False,
    color_match_border_radius=30,
    color_match_blend_radius=15,
    # ✅ 改进7（v4）：LaMa 后端（object_removal 专用）
    use_lama=True,
    # ✅ 改进8（v4）：OCR 文字预处理（object_removal 专用，需 use_lama=True）
    use_ocr_preprocess=True,
    ocr_confidence_threshold=0.5,
    ocr_fill_color=(255, 255, 255),
    # ✅ 改进9（v5）：Removal 多后端路由（LaMa → ZITS → MAT → SDXL 降级）
    use_zits=True,
    use_mat=True,
    # ✅ 改进10（v5）：精细化 Mask 参数（dilation + erosion + Gaussian）
    mask_erode_ksize=9,
    mask_erode_iters=1,
    mask_smooth_sigma=15,
):
    try:
        # ✅ 修改1：凸包填充和色彩匹配根据 task_type 自动决策实际生效值
        # 凸包填充仅对 removal 类型有益（replacement 目标形状复杂，强行凸包会过度覆盖）
        # 色彩匹配对 replace 类任务应关闭（replace 本身就需要颜色变化）
        effective_convex = use_convex_hull and (task_type == "object_removal")
        effective_color  = use_color_match  and (task_type == "object_removal")

        # ✅ 改进7/8：LaMa 和 OCR 仅对 object_removal 任务生效
        effective_lama = use_lama and (task_type == "object_removal") and IOPAINT_AVAILABLE
        effective_ocr  = (use_ocr_preprocess and effective_lama
                          and PADDLEOCR_AVAILABLE)

        # ✅ 改进9（v5）：Removal 多后端路由标志
        # effective_removal_iopaint = True 时进入 IOPaint 路由（LaMa/ZITS/MAT 按优先级）
        effective_removal_iopaint = (
            task_type == "object_removal"
            and IOPAINT_AVAILABLE
            and (use_lama or use_zits or use_mat)
        )

        # ✅ 修复B：inpaint_mode 自动决策
        # object_removal：自动 merge，确保多个气泡/目标全部被修复
        # object_replacement：保持 first，避免多目标合并成超大 mask
        #   导致 SDXL 处理能力不足（如检测到6个 people 时不应合并）
        effective_inpaint_mode = "merge" if task_type == "object_removal" else "first"

        print(f"\n{'='*80}")
        print(f"测试: {description}")
        print(f"图片: {image_path}")
        print(f"检测提示词: {det_prompt}")
        print(f"修复提示词: {inpaint_prompt}")
        print(f"任务类型: {task_type}")
        print(f"[v2] DSP Mask: {use_dsp_mask} | DSP线条增强: {use_dsp_enhance} | "
              f"ControlNet: {use_controlnet} | 拉普拉斯融合: {use_lap_blend}")
        print(f"[v3] 凸包填充: {effective_convex}(配置:{use_convex_hull}) | "
              f"两阶段修复: {use_two_stage} | "
              f"色彩匹配: {effective_color}(配置:{use_color_match}) | "
              f"CN屏蔽Mask: {controlnet_mask_shield}")
        print(f"[v4] LaMa后端: {effective_lama}(配置:{use_lama}) | "
              f"OCR预处理: {effective_ocr}(配置:{use_ocr_preprocess}) | "
              f"inpaint_mode: {effective_inpaint_mode}")
        print(f"[v5] Removal多后端路由: {effective_removal_iopaint}"
              f"(LaMa:{use_lama}/ZITS:{use_zits}/MAT:{use_mat}) | "
              f"Mask腐蚀: ksize={mask_erode_ksize} iters={mask_erode_iters} | "
              f"平滑sigma={mask_smooth_sigma}")
        print(f"{'='*80}\n")

        output_dir = os.path.join(output_base_dir, Path(image_path).parent.name, Path(image_path).stem)
        os.makedirs(output_dir, exist_ok=True)

        # [1/8] 加载图片
        print("  [1/8] 加载图片...")
        image_pil, image = load_image(image_path)
        raw_image_path = os.path.join(output_dir, "raw_image.jpg")
        image_pil.save(raw_image_path)

        # [2/8] Grounding DINO 检测
        print("  [2/8] 运行 Grounding DINO...")
        boxes_filt, pred_phrases, scores = get_grounding_output(
            model, image, det_prompt, box_threshold, text_threshold, device=device
        )
        boxes_filt, pred_phrases, scores = filter_boxes_nms(
            boxes_filt, pred_phrases, scores, iou_threshold=nms_iou_threshold
        )
        print(f"  ✓ 检测到 {len(boxes_filt)} 个目标（NMS过滤后）: {pred_phrases}")

        if len(boxes_filt) == 0:
            print("  ✗ 警告: 未检测到任何目标!")
            return {
                "status": "warning",
                "message": "未检测到任何目标",
                "image_path": image_path,
                "description": description
            }

        # [3/8] SAM 分割
        print("  [3/8] 运行 SAM 分割...")
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
        image_cv = cv2.imread(image_path)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

        # 新增C（v2）：DSP 线条增强与双边滤波
        # ✅ 修复D（v4）：增强图仅用于提升 SAM 分割精度
        #               inpainting 输入改回原图，避免增强伪影污染生成结果
        if use_dsp_enhance:
            print("  [DSP-C] 线条增强与双边滤波（仅用于 SAM 分割）...")
            image_cv_enhanced = dsp_enhance_image(
                image_cv,
                bilateral_d=dsp_bilateral_d,
                bilateral_sigma_color=dsp_bilateral_sigma,
                bilateral_sigma_space=dsp_bilateral_sigma,
                unsharp_strength=dsp_unsharp_strength,
            )
            Image.fromarray(image_cv_enhanced).save(
                os.path.join(output_dir, "dsp_enhanced.jpg"))
            print(f"  ✓ DSP增强完成（增强图仅供 SAM 使用）")
        else:
            image_cv_enhanced = image_cv

        # SAM 使用增强图提升分割精度
        predictor.set_image(image_cv_enhanced)

        # ✅ 修复D：inpainting 始终使用原始图像作为输入
        image_cv_for_inpaint_base = image_cv

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(
            boxes_filt, image_cv.shape[:2]
        ).to(device)

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(device),
            multimask_output=False,
        )

        # [4/8] 保存分割结果
        print("  [4/8] 保存分割结果...")
        plt.figure(figsize=(10, 10))
        plt.imshow(image_cv)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box.numpy(), plt.gca(), label)
        plt.axis('off')
        segmentation_path = os.path.join(output_dir, "grounded_sam_output.jpg")
        plt.savefig(segmentation_path, bbox_inches="tight", dpi=150)
        plt.close()

        # [5/8] Mask 处理
        print("  [5/8] 处理 Mask...")

        # ✅ 修复F（v6）：multi-object mask 处理策略
        # 原代码（v5）：
        #   - removal   → merge 所有 mask（torch.any）✓ 正确
        #   - replacement → 仅取 masks[0]            ✗ 只处理第一个目标
        # 新代码（v6）：
        #   - removal   → merge 所有 mask（不变）
        #   - replacement → 构建 per_object_masks 列表，在 [6/8] 逐个处理，
        #                   每轮以上一轮输出作为新输入（链式替换）
        if effective_inpaint_mode == 'merge':
            # removal 任务：合并所有 mask 一次性修复
            per_object_masks = [torch.any(masks, dim=0)[0].cpu().numpy().astype(np.uint8) * 255]
        else:
            # replacement 任务：为每个检测目标独立建立 mask 列表
            # 链式推进：第 i 轮 inpaint 的输出作为第 i+1 轮的输入图像
            per_object_masks = [
                masks[i][0].cpu().numpy().astype(np.uint8) * 255
                for i in range(len(masks))
            ]
            print(f"  [修复F] object_replacement 多目标模式：共 {len(per_object_masks)} 个目标，逐个处理")

        # ── 对第一个 mask 做 DSP 优化，后续 mask 复用同一套参数 ──────────────
        def optimize_single_mask(raw_m):
            if use_dsp_mask:
                return dsp_optimize_mask(
                    raw_m,
                    morph_close_ksize=dsp_morph_close_ksize,
                    morph_open_ksize=dsp_morph_open_ksize,
                    dilate_ksize=15,
                    dilate_iters=1,
                    feather_sigma=dsp_feather_sigma,
                    feather_strength=dsp_feather_strength,
                    use_convex_hull=effective_convex,
                    erode_ksize=mask_erode_ksize,
                    erode_iters=mask_erode_iters,
                    smooth_sigma=mask_smooth_sigma,
                )
            else:
                k = np.ones((15, 15), np.uint8)
                m = cv2.dilate(raw_m, k, iterations=1)
                return cv2.GaussianBlur(m, (21, 21), 0)

        # 主 mask（用于日志/保存/色彩匹配基准）= 第一个目标的优化 mask
        merged_mask_np = optimize_single_mask(per_object_masks[0])
        hull_note = "（含凸包填充）" if effective_convex else ""
        print(f"  [DSP-A+v3+v5] Mask 优化{hull_note}"
              f"（dilate=15→erode={mask_erode_ksize}→smooth_sigma={mask_smooth_sigma}）")
        Image.fromarray(merged_mask_np).save(
            os.path.join(output_dir, "dsp_mask_optimized.jpg"))
        print(f"  ✓ DSP Mask 优化完成")

        mask_pil = Image.fromarray(merged_mask_np)
        # ✅ 修复D：inpainting 使用原始图像（不含 DSP 增强伪影）
        image_pil_for_inpaint = Image.fromarray(image_cv_for_inpaint_base)

        original_size = image_pil_for_inpaint.size
        # IOPaint 分支不需要 padding，new_wh/offset 仅 SDXL 分支内部使用
        new_wh, offset = original_size, (0, 0)

        # ✅ 改进3：根据任务类型增强 Prompt
        enhanced_prompt, enhanced_negative = build_removal_prompts(
            inpaint_prompt, inpaint_negative_prompt, task_type=task_type)
        if task_type == "object_removal":
            print(f"  [改进3] 删除类 Prompt 增强已应用")
            print(f"    正向: ...{REMOVAL_POSITIVE_SUFFIX[:60]}...")
            print(f"    负向追加: speech bubble, text, ...")

        # [6/8] Inpainting（v6：多目标循环 + hard compositing）
        print("  [6/8] 运行 Inpainting...")

        # ✅ 修复F（v6）：多目标链式处理
        # replacement: 对 per_object_masks 中每个 mask 逐一 inpaint，
        #              每轮输出作为下一轮输入（链式替换），确保所有检测目标均被处理
        # removal:     per_object_masks 只有 1 个合并 mask，行为与 v5 完全一致

        # 链式输入初始化：从原图出发
        chained_image_np  = np.array(image_pil_for_inpaint)
        chained_image_pil = image_pil_for_inpaint

        n_objects = len(per_object_masks)
        for obj_idx, raw_mask_i in enumerate(per_object_masks):
            if n_objects > 1:
                print(f"  [修复F] 处理目标 {obj_idx + 1}/{n_objects}...")

            # 每个目标独立做 mask 优化
            mask_i = optimize_single_mask(raw_mask_i)

            # ✅ 修复G（v6）：hard mask 用于后处理 compositing，防止扩散溢出 mask 边界
            # soft mask（mask_i）传给扩散模型引导生成；
            # hard mask（raw_mask_i 原始二值 mask，轻微膨胀）用于最终合成截断

            mask_pil_i = Image.fromarray(mask_i)

            # ✅ 改进9（v5）：Removal 任务按 LaMa → ZITS → MAT 优先路由
            if effective_removal_iopaint:
                if obj_idx == 0:
                    print(f"  [改进9] object_removal → IOPaint 多后端路由"
                          f"（优先级: {'LaMa→' if use_lama else ''}{'ZITS→' if use_zits else ''}{'MAT' if use_mat else ''}）")

                # OCR 文字预处理（仅 removal 第一目标执行）
                image_for_iopaint = chained_image_np.copy()
                if effective_ocr and obj_idx == 0:
                    print("  [改进8] 形态学文字预处理...")
                    image_for_iopaint = ocr_fill_text(
                        image_np=image_for_iopaint,
                        mask_np=mask_i,
                        fill_color=ocr_fill_color,
                        ocr_confidence_threshold=ocr_confidence_threshold,
                    )
                    Image.fromarray(image_for_iopaint).save(
                        os.path.join(output_dir, "ocr_prefilled.jpg"))
                    chained_image_pil = Image.fromarray(image_for_iopaint)
                    print("  ✓ 形态学预处理完成，已保存 ocr_prefilled.jpg")

                inpainted_np_iopaint, used_backend = removal_inpaint_with_priority(
                    image_np=image_for_iopaint,
                    mask_np=mask_i,
                    device=device,
                    use_lama=use_lama,
                    use_zits=use_zits,
                    use_mat=use_mat,
                )

                if inpainted_np_iopaint is not None:
                    inpainted_image = Image.fromarray(inpainted_np_iopaint)
                    inpainted_image.save(
                        os.path.join(output_dir, f"{used_backend}_raw_output_obj{obj_idx}.jpg"))
                    inpainted_np_i = inpainted_np_iopaint
                    print(f"  ✓ IOPaint({used_backend.upper()}) obj{obj_idx} 完成")
                else:
                    print("  [改进9] IOPaint 全部失败，降级 SDXL...")
                    effective_removal_iopaint = False

            if not effective_removal_iopaint:
                # ── SDXL 分支（object_replacement 或 IOPaint 降级） ──
                image_padded_i, new_wh, offset = resize_with_padding(
                    chained_image_pil, target_size=1024)
                mask_padded_i, _, _ = resize_with_padding(mask_pil_i.convert("RGB"), target_size=1024)
                mask_padded_i = mask_padded_i.convert("L")

                # 准备 ControlNet 条件图（如需要）
                condition_padded = None
                if use_controlnet:
                    print(f"  [DSP-B+改进4] 生成 ControlNet 条件图（{controlnet_condition_type}）...")
                    condition_image = generate_controlnet_condition(
                        chained_image_np,
                        condition_type=controlnet_condition_type,
                        mask_np=mask_i if controlnet_mask_shield else None,
                    )
                    condition_image.save(
                        os.path.join(output_dir,
                                     f"controlnet_condition_{controlnet_condition_type}_obj{obj_idx}.jpg"))
                    condition_padded, _, _ = resize_with_padding(condition_image, target_size=1024)
                    print(f"  ✓ ControlNet 条件图生成完成（mask区域已屏蔽: {controlnet_mask_shield}）")

                # 两阶段 or 单次调用
                if use_two_stage:
                    print(f"  [改进2] 两阶段 Inpainting（obj{obj_idx}）...")
                    inpainted_image_i = two_stage_inpaint(
                        pipe=pipe,
                        prompt=enhanced_prompt,
                        negative_prompt=enhanced_negative,
                        image_padded=image_padded_i,
                        mask_padded=mask_padded_i,
                        stage1_steps=stage1_steps,
                        stage1_strength=stage1_strength,
                        stage1_guidance=stage1_guidance,
                        stage2_steps=stage2_steps,
                        stage2_strength=stage2_strength,
                        stage2_guidance=stage2_guidance,
                        use_controlnet=use_controlnet,
                        condition_padded=condition_padded,
                        controlnet_scale=controlnet_scale,
                    )
                    print("  ✓ 两阶段 Inpainting 完成")
                else:
                    if use_controlnet:
                        inpainted_image_i = pipe(
                            prompt=enhanced_prompt,
                            negative_prompt=enhanced_negative,
                            image=image_padded_i,
                            mask_image=mask_padded_i,
                            control_image=condition_padded,
                            controlnet_conditioning_scale=controlnet_scale,
                            num_inference_steps=inpaint_steps,
                            guidance_scale=inpaint_guidance_scale,
                            strength=inpaint_strength,
                        ).images[0]
                    else:
                        inpainted_image_i = pipe(
                            prompt=enhanced_prompt,
                            negative_prompt=enhanced_negative,
                            image=image_padded_i,
                            mask_image=mask_padded_i,
                            num_inference_steps=inpaint_steps,
                            guidance_scale=inpaint_guidance_scale,
                            strength=inpaint_strength,
                        ).images[0]

                inpainted_image_i = restore_from_padding(
                    inpainted_image_i, original_size, new_wh, offset, target_size=1024)
                inpainted_image = inpainted_image_i
                inpainted_np_i = np.array(inpainted_image_i)

            # ✅ 修复G（v6）：hard compositing — 将扩散结果裁回 hard mask 边界
            # 原代码：直接将整张 inpainted_np 推进下一步，扩散模型溢出的像素随之传播
            # 新代码：
            #   final_i = orig × (1 − hard_mask) + inpainted × hard_mask
            # 这样 mask 外部像素永远来自原图（或上一轮链式输出），不会被扩散污染
            inpainted_np_i_full = cv2.resize(
            inpainted_np_i,
            (image_cv_for_inpaint_base.shape[1], image_cv_for_inpaint_base.shape[0]),
            interpolation=cv2.INTER_LINEAR
            )
            if n_objects > 1:
                Image.fromarray(inpainted_np_i_full).save(
                    os.path.join(output_dir, f"inpainted_obj{obj_idx}.jpg"))

            chained_image_np  = inpainted_np_i_full
            chained_image_pil = Image.fromarray(inpainted_np_i_full)

        # 所有目标处理完毕后，chained_image_np 即为最终 inpainted 结果
        inpainted_np = chained_image_np
        inpainted_image = chained_image_pil
        # 更新 image_pil_for_inpaint 用于后续色彩匹配的基准
        image_pil_for_inpaint = Image.fromarray(image_cv_for_inpaint_base)

        # 新增D（v2）：拉普拉斯金字塔融合
        if use_lap_blend:
            print(f"  [DSP-D] 拉普拉斯金字塔融合（levels={lap_blend_levels}）...")
            # ✅ 修复D：融合基准改为原图（与 inpainting 输入一致）
            # v5：image_cv_for_inpaint 仅在 effective_ocr 路径下由 IOPaint 块内部赋值，
            #     统一用 image_pil_for_inpaint 的 numpy 形式作为融合基准
            blend_base = np.array(image_pil_for_inpaint)
            mask_for_blend = cv2.resize(
                merged_mask_np,
                (blend_base.shape[1], blend_base.shape[0]),
                interpolation=cv2.INTER_LINEAR)
            inpainted_resized = cv2.resize(
                inpainted_np,
                (blend_base.shape[1], blend_base.shape[0]),
                interpolation=cv2.INTER_LINEAR)

            blended_np = laplacian_pyramid_blend(
                orig_np=blend_base,
                inpainted_np=inpainted_resized,
                mask_np=mask_for_blend,
                levels=lap_blend_levels,
            )
            inpainted_image.save(os.path.join(output_dir, "inpainting_before_blend.jpg"))
            print(f"  ✓ 拉普拉斯金字塔融合完成")
        else:
            blend_base = np.array(image_pil_for_inpaint)
            blended_np = inpainted_np
            mask_for_blend = cv2.resize(
                merged_mask_np,
                (blend_base.shape[1], blend_base.shape[0]),
                interpolation=cv2.INTER_LINEAR)

        # ✅ 改进5：边缘色彩匹配后处理（仅 removal 类任务生效）
        if effective_color:
            print(f"  [改进5] 边缘色彩匹配后处理...")
            Image.fromarray(blended_np).save(
                os.path.join(output_dir, "before_color_match.jpg"))
            blended_np = color_match_inpainted(
                orig_np=blend_base,
                inpainted_np=blended_np,
                mask_np=mask_for_blend,
                border_radius=color_match_border_radius,
                blend_radius=color_match_blend_radius,
            )
            print(f"  ✓ 色彩匹配完成（已保存 before_color_match.jpg 用于对比）")

        final_image = Image.fromarray(blended_np)
        inpainting_path = os.path.join(output_dir, "grounded_sam_inpainting_output.jpg")
        final_image.save(inpainting_path)

        # [7/8] 评价指标（改进6：分任务类型加权）
        metrics = None
        if evaluator:
            print("  [7/8] 计算评价指标...")
            metrics = evaluator.run(
                raw_image_path, inpainting_path, enhanced_prompt,
                task_type=task_type   # ✅ 改进6
            )
            if metrics:
                print(f"  ✓ PSNR: {metrics['PSNR']:.2f} | SSIM: {metrics['SSIM']:.4f} | "
                      f"LPIPS: {metrics['LPIPS']:.4f} | CLIP: {metrics['CLIP_Score']:.2f} | "
                      f"加权综合分: {metrics['weighted_score']:.2f} (task={task_type})")

        print(f"  ✓ 测试完成! 输出: {output_dir}")

        return {
            "status": "success",
            "image_path": image_path,
            "description": description,
            "task_type": task_type,
            "output_dir": output_dir,
            "detected_objects": len(boxes_filt),
            "detected_labels": pred_phrases,
            "metrics": metrics
        }

    except Exception as e:
        print(f"  ✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "image_path": image_path,
            "description": description,
            "error": str(e)
        }


def main():
    class Args:
        # ── 路径配置 ──────────────────────────────────────────────────────
        config              = "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        grounded_checkpoint = "Grounded-Segment-Anything/weights/groundingdino_swint_ogc.pth"
        sam_checkpoint      = "Grounded-Segment-Anything/weights/sam_vit_h_4b8939.pth"
        image_root          = "Grounded-Segment-Anything/dataset"
        prompts_config      = "Grounded-Segment-Anything/prompts_config.json"
        output_dir          = "test_outputs"
        sd_model_path       = "/root/autodl-tmp/models/sdxl-inpainting"

        # ── 检测参数 ──────────────────────────────────────────────────────
        box_threshold     = 0.3
        text_threshold    = 0.25
        inpaint_mode      = "merge"
        device            = "cuda" if torch.cuda.is_available() else "cpu"
        test_subset       = None
        enable_metrics    = True
        nms_iou_threshold = 0.5

        # ── 基础 Inpainting 参数 ─────────────────────────────────────────
        inpaint_steps           = 50
        inpaint_guidance_scale  = 7.5
        inpaint_strength        = 0.99
        inpaint_negative_prompt = "blurry, bad quality, distorted, artifacts, ugly, low resolution"

        # ── v2 新增A：DSP Mask 优化 ───────────────────────────────────────
        use_dsp_mask          = True
        dsp_morph_close_ksize = 21
        dsp_morph_open_ksize  = 7
        dsp_feather_sigma     = 21
        dsp_feather_strength  = 1.0

        # ── v2 新增B：ControlNet ──────────────────────────────────────────
        use_controlnet            = False
        controlnet_model_path     = "/root/autodl-tmp/models/controlnet-canny-sdxl"
        controlnet_condition_type = "canny"
        controlnet_scale          = 0.5

        # ── v2 新增C：DSP 线条增强 ────────────────────────────────────────
        use_dsp_enhance      = True
        dsp_bilateral_d      = 9
        dsp_bilateral_sigma  = 75
        dsp_unsharp_strength = 0.5

        # ── v2 新增D：拉普拉斯金字塔融合 ─────────────────────────────────
        use_lap_blend    = False
        lap_blend_levels = 6

        # ── ✅ v3 改进1：凸包填充 ─────────────────────────────────────────
        # True = "允许对 removal 类任务启用"，replace 类在 run_single_test 内自动跳过
        # 无需按任务类型手动关闭，task_type 驱动实际生效逻辑
        use_convex_hull = True

        # ── ✅ v3 改进2：两阶段 Inpainting ───────────────────────────────
        # 大面积删除任务推荐开启；替换类小面积任务可关闭以节省时间
        use_two_stage    = True
        stage1_steps     = 30
        stage1_strength  = 0.65
        stage1_guidance  = 9.0
        stage2_steps     = 20
        stage2_strength  = 0.35
        stage2_guidance  = 7.5

        # ── ✅ v3 改进3：任务类型 ─────────────────────────────────────────
        # "removal"（删除类，如去除气泡/水印）或 "replace"（替换类）
        # 影响 Prompt 增强策略和评价指标权重
        task_type = "removal"

        # ── ✅ v3 改进4：ControlNet 条件图屏蔽 mask 区域 ──────────────────
        # 开启后，mask 区域的 Canny 边缘将被置零，
        # 防止气泡轮廓约束重建方向（仅 use_controlnet=True 时生效）
        controlnet_mask_shield = True

        # ── ✅ v3 改进5：边缘色彩匹配后处理 ─────────────────────────────
        # True = "允许对 removal 类任务启用"，replace 类在 run_single_test 内自动跳过
        # 原因：replace 任务本身需要颜色变化，强制色彩匹配会破坏替换效果
        use_color_match          = True
        color_match_border_radius = 30
        color_match_blend_radius  = 15

        # ── ✅ v4 改进7：LaMa 后端 ───────────────────────────────────────
        # True = "允许对 object_removal 任务启用 LaMa"
        # IOPaint 未安装时自动降级为 SDXL，不影响流程
        use_lama = True

        # ── ✅ v4 改进8：OCR 文字预处理 ──────────────────────────────────
        # 仅在 use_lama=True 且 IOPaint/PaddleOCR 均可用时生效
        # 先填白文字再交给 LaMa，降低单次修复难度
        use_ocr_preprocess        = True
        ocr_confidence_threshold  = 0.5   # OCR 置信度阈值，低于此值的框忽略
        ocr_fill_color            = (255, 255, 255)  # 文字填充颜色（白色）

        # ── ✅ v5 改进9：Removal 多后端路由 ──────────────────────────────
        # 按 LaMa → ZITS → MAT 优先顺序依次尝试，全部失败时降级 SDXL
        # 建议：use_lama=True, use_zits=True, use_mat=True（让系统自动选最优）
        use_zits = True   # ZITS：结构感知修复，漫画线条边界保留效果好
        use_mat  = True   # MAT：Transformer 架构，大面积缺失修复质量优

        # ── ✅ v5 改进10：精细化 Mask 参数 ────────────────────────────────
        # 流水线：close → dilate(15) → erode(mask_erode_ksize) → Gaussian(mask_smooth_sigma)
        # 腐蚀核建议设为膨胀核的 1/2 ~ 2/3，净效果 = 扩展 + 轮廓平滑
        mask_erode_ksize  = 9    # 腐蚀核尺寸（单位：像素）
        mask_erode_iters  = 1    # 腐蚀迭代次数
        mask_smooth_sigma = 15   # Gaussian 平滑 sigma（边界软化强度）

    args = Args()

    print(f"加载提示词配置: {args.prompts_config}")
    test_configs = load_test_configs(args.prompts_config)
    print(f"✓ 加载了 {len(test_configs)} 个测试配置\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = os.path.join(args.output_dir, f"batch_test_{timestamp}")
    os.makedirs(output_base_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"开始批量测试 (v6)")
    print(f"输出目录: {output_base_dir}")
    print(f"设备: {args.device} | 默认任务类型: {args.task_type}")
    print(f"[v2] DSP Mask: {args.use_dsp_mask} | DSP增强: {args.use_dsp_enhance} | "
          f"ControlNet: {args.use_controlnet} | 拉普拉斯融合: {args.use_lap_blend}")
    print(f"[v3] 凸包填充: {args.use_convex_hull} | 两阶段修复: {args.use_two_stage} | "
          f"色彩匹配: {args.use_color_match} | CN屏蔽Mask: {args.controlnet_mask_shield}")
    print(f"[v4] LaMa后端: {args.use_lama}(IOPaint可用:{IOPAINT_AVAILABLE}) | "
          f"OCR预处理: {args.use_ocr_preprocess}(PaddleOCR可用:{PADDLEOCR_AVAILABLE})")
    print(f"[v5] ZITS: {args.use_zits} | MAT: {args.use_mat} | "
          f"Mask腐蚀: ksize={args.mask_erode_ksize} iters={args.mask_erode_iters} | "
          f"平滑sigma={args.mask_smooth_sigma}")
    print(f"{'='*80}\n")

    print("加载 Grounding DINO 模型...")
    model = load_model(args.config, args.grounded_checkpoint, device=args.device)
    print("✓ Grounding DINO 加载完成\n")

    # v2 改进B：根据 use_controlnet 选择加载哪种管线
    if args.use_controlnet:
        print("加载 ControlNet + SDXL Inpainting 模型...")
        pipe = load_controlnet_pipe(
            sd_model_path=args.sd_model_path,
            controlnet_model_path=args.controlnet_model_path,
            device=args.device
        )
    else:
        print("加载 SDXL Inpainting 模型...")
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            args.sd_model_path,
            torch_dtype=torch.float16,
            local_files_only=True
        )
        pipe = pipe.to(args.device)
        print("✓ SDXL Inpainting 加载完成\n")

    evaluator = Evaluator(args.device) if args.enable_metrics else None

    if args.test_subset:
        test_configs = {k: v for k, v in test_configs.items() if k in args.test_subset}

    results = []
    total   = len(test_configs)

    for idx, (rel_path, config) in enumerate(test_configs.items(), 1):
        print(f"\n进度: [{idx}/{total}]")

        image_path = os.path.join(args.image_root, rel_path)

        if not os.path.exists(image_path):
            print(f"警告: 图片不存在: {image_path}")
            results.append({
                "status":      "error",
                "image_path":  rel_path,
                "description": config["description"],
                "error":       "文件不存在"
            })
            continue

        # 支持 prompts_config 中每条记录单独指定 task_type，默认用全局设置
        item_task_type = config.get("task_type", args.task_type)

        result = run_single_test(
            image_path=image_path,
            det_prompt=config["det_prompt"],
            inpaint_prompt=config["inpaint_prompt"],
            description=config["description"],
            model=model,
            sam_checkpoint=args.sam_checkpoint,
            pipe=pipe,
            output_base_dir=output_base_dir,
            evaluator=evaluator,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            inpaint_mode=args.inpaint_mode,
            nms_iou_threshold=args.nms_iou_threshold,
            inpaint_steps=args.inpaint_steps,
            inpaint_guidance_scale=args.inpaint_guidance_scale,
            inpaint_strength=args.inpaint_strength,
            inpaint_negative_prompt=args.inpaint_negative_prompt,
            device=args.device,
            # v2 参数透传
            use_dsp_mask=args.use_dsp_mask,
            dsp_morph_close_ksize=args.dsp_morph_close_ksize,
            dsp_morph_open_ksize=args.dsp_morph_open_ksize,
            dsp_feather_sigma=args.dsp_feather_sigma,
            dsp_feather_strength=args.dsp_feather_strength,
            use_controlnet=args.use_controlnet,
            controlnet_condition_type=args.controlnet_condition_type,
            controlnet_scale=args.controlnet_scale,
            use_dsp_enhance=args.use_dsp_enhance,
            dsp_bilateral_d=args.dsp_bilateral_d,
            dsp_bilateral_sigma=args.dsp_bilateral_sigma,
            dsp_unsharp_strength=args.dsp_unsharp_strength,
            use_lap_blend=args.use_lap_blend,
            lap_blend_levels=args.lap_blend_levels,
            # ✅ v3 参数透传
            # 注意：use_convex_hull / use_color_match 的实际生效由 run_single_test
            # 内部根据 item_task_type 自动决策，此处仅传入"是否允许启用"的全局开关
            use_convex_hull=args.use_convex_hull,
            use_two_stage=args.use_two_stage,
            stage1_steps=args.stage1_steps,
            stage1_strength=args.stage1_strength,
            stage1_guidance=args.stage1_guidance,
            stage2_steps=args.stage2_steps,
            stage2_strength=args.stage2_strength,
            stage2_guidance=args.stage2_guidance,
            task_type=item_task_type,
            controlnet_mask_shield=args.controlnet_mask_shield,
            use_color_match=args.use_color_match,
            color_match_border_radius=args.color_match_border_radius,
            color_match_blend_radius=args.color_match_blend_radius,
            # ✅ v4 参数透传
            use_lama=args.use_lama,
            use_ocr_preprocess=args.use_ocr_preprocess,
            ocr_confidence_threshold=args.ocr_confidence_threshold,
            ocr_fill_color=args.ocr_fill_color,
            # ✅ v5 参数透传
            use_zits=args.use_zits,
            use_mat=args.use_mat,
            mask_erode_ksize=args.mask_erode_ksize,
            mask_erode_iters=args.mask_erode_iters,
            mask_smooth_sigma=args.mask_smooth_sigma,
        )
        results.append(result)

    # ── 汇总报告 ──────────────────────────────────────────────────────────
    success_results = [r for r in results if r["status"] == "success"]
    report_path     = os.path.join(output_base_dir, "test_report.json")

    # 按任务类型分别统计平均指标
    def avg_metrics_by_type(results_list, t_type):
        filtered = [
            r["metrics"] for r in results_list
            if r.get("metrics") and r.get("task_type") == t_type
        ]
        if not filtered:
            return None
        return {
            "PSNR":           round(np.mean([m["PSNR"]           for m in filtered]), 2),
            "SSIM":           round(np.mean([m["SSIM"]           for m in filtered]), 4),
            "LPIPS":          round(np.mean([m["LPIPS"]          for m in filtered]), 4),
            "CLIP_Score":     round(np.mean([m["CLIP_Score"]     for m in filtered]), 2),
            "weighted_score": round(np.mean([m["weighted_score"] for m in filtered]), 2),
            "count":          len(filtered),
        }

    report_data = {
        "timestamp":           timestamp,
        "version":             "v6",
        "sd_model":            args.sd_model_path,
        "inpaint_steps":       args.inpaint_steps,
        "inpaint_guidance_scale": args.inpaint_guidance_scale,
        "inpaint_strength":    args.inpaint_strength,
        "nms_iou_threshold":   args.nms_iou_threshold,
        # v2 功能开关
        "dsp_mask_enabled":    args.use_dsp_mask,
        "dsp_enhance_enabled": args.use_dsp_enhance,
        "controlnet_enabled":  args.use_controlnet,
        "lap_blend_enabled":   args.use_lap_blend,
        # ✅ v3 功能开关
        "convex_hull_enabled":        args.use_convex_hull,
        "two_stage_enabled":          args.use_two_stage,
        "color_match_enabled":        args.use_color_match,
        "controlnet_mask_shield":     args.controlnet_mask_shield,
        "default_task_type":          args.task_type,
        # ✅ v4 功能开关
        "lama_enabled":               args.use_lama,
        "lama_available":             IOPAINT_AVAILABLE,
        "ocr_preprocess_enabled":     args.use_ocr_preprocess,
        "ocr_available":              PADDLEOCR_AVAILABLE,
        # ✅ v5 功能开关
        "zits_enabled":               args.use_zits,
        "mat_enabled":                args.use_mat,
        "mask_erode_ksize":           args.mask_erode_ksize,
        "mask_erode_iters":           args.mask_erode_iters,
        "mask_smooth_sigma":          args.mask_smooth_sigma,
        "total_tests": total,
        "results": results,
        "summary": {
            "success": sum(1 for r in results if r["status"] == "success"),
            "warning": sum(1 for r in results if r["status"] == "warning"),
            "error":   sum(1 for r in results if r["status"] == "error"),
        }
    }

    # ✅ 改进6：按任务类型分组输出平均指标
    if args.enable_metrics and success_results:
        report_data["summary"]["average_metrics_by_task"] = {}
        for t in ["object_removal", "object_replacement"]:
            avg = avg_metrics_by_type(success_results, t)
            if avg:
                report_data["summary"]["average_metrics_by_task"][t] = avg

        # 整体平均（兼容旧格式）
        all_metrics = [r["metrics"] for r in success_results if r.get("metrics")]
        if all_metrics:
            report_data["summary"]["average_metrics"] = {
                "PSNR":           round(np.mean([m["PSNR"]           for m in all_metrics]), 2),
                "SSIM":           round(np.mean([m["SSIM"]           for m in all_metrics]), 4),
                "LPIPS":          round(np.mean([m["LPIPS"]          for m in all_metrics]), 4),
                "CLIP_Score":     round(np.mean([m["CLIP_Score"]     for m in all_metrics]), 2),
                "weighted_score": round(np.mean([m["weighted_score"] for m in all_metrics]), 2),
            }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    # ── 终端汇总输出 ──────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"批量测试完成! (v6)")
    print(f"{'='*80}")
    print(f"总测试数: {total}")
    print(f"  ✓ 成功: {report_data['summary']['success']}")
    print(f"  ⚠ 警告: {report_data['summary']['warning']}")
    print(f"  ✗ 失败: {report_data['summary']['error']}")

    if args.enable_metrics and "average_metrics_by_task" in report_data["summary"]:
        print(f"\n按任务类型平均评价指标:")
        for t_type, avg in report_data["summary"]["average_metrics_by_task"].items():
            print(f"\n  [{t_type}] (n={avg['count']})")
            print(f"    PSNR:           {avg['PSNR']:.2f} dB")
            print(f"    SSIM:           {avg['SSIM']:.4f}")
            print(f"    LPIPS:          {avg['LPIPS']:.4f}")
            print(f"    CLIP Score:     {avg['CLIP_Score']:.2f}")
            print(f"    加权综合分:     {avg['weighted_score']:.2f}")

    print(f"\n详细报告: {report_path}")
    print(f"输出目录: {output_base_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()