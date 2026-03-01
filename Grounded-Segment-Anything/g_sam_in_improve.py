"""
批量测试 Grounded-SAM Inpainting（支持评价指标）
已同步 grounded_sam_inpainting_improved.py 的全部6项改动

新增功能（v2）：
  ✅ 新增A：DSP Mask 优化（平滑、形态学修补、羽化）
  ✅ 新增B：ControlNet 约束下的 SD Inpainting
  ✅ 新增C：DSP 线条增强与双边滤波（预处理图像质量）
  ✅ 新增D：拉普拉斯金字塔融合（边缘无缝合成）
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
    # ✅ 新增B：导入 ControlNet 相关模块
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


# ==================== ✅ 新增A：DSP Mask 优化模块 ====================
# 原代码：仅做简单膨胀+高斯模糊
# 新代码：增加形态学修补（闭运算填孔）+ 平滑轮廓 + 可控羽化强度

def dsp_optimize_mask(mask_np: np.ndarray,
                      morph_close_ksize: int = 21,
                      morph_open_ksize: int = 7,
                      dilate_ksize: int = 15,
                      dilate_iters: int = 1,
                      feather_sigma: int = 21,
                      feather_strength: float = 1.0) -> np.ndarray:
    """
    DSP Mask 优化：形态学修补 → 膨胀 → 羽化
    Args:
        mask_np       : uint8 二值 mask (0 or 255)
        morph_close_ksize: 闭运算核大小，填补 mask 内部空洞
        morph_open_ksize : 开运算核大小，去除细碎噪点
        dilate_ksize  : 膨胀核大小
        dilate_iters  : 膨胀迭代次数
        feather_sigma : 高斯羽化 sigma（必须为奇数）
        feather_strength: 羽化混合强度 [0,1]，1=完全羽化，0=保持二值
    Returns:
        处理后的 uint8 mask
    """
    # Step 1：闭运算（填补 mask 内部空洞/碎裂区域）
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_close_ksize, morph_close_ksize))
    mask_closed = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel_close)

    # Step 2：开运算（去除细碎噪点，平滑轮廓）
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_open_ksize, morph_open_ksize))
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_open)

    # Step 3：膨胀（扩展 mask 边界，覆盖目标周围区域）
    kernel_dilate = np.ones((dilate_ksize, dilate_ksize), np.uint8)
    mask_dilated = cv2.dilate(mask_opened, kernel_dilate, iterations=dilate_iters)

    # Step 4：高斯羽化（软化边缘，避免合成硬边）
    feather_ksize = feather_sigma if feather_sigma % 2 == 1 else feather_sigma + 1
    mask_feathered = cv2.GaussianBlur(mask_dilated, (feather_ksize, feather_ksize), 0)

    # Step 5：按 feather_strength 混合羽化结果与硬边 mask
    if feather_strength < 1.0:
        mask_out = cv2.addWeighted(
            mask_feathered, feather_strength,
            mask_dilated, 1.0 - feather_strength,
            0
        )
    else:
        mask_out = mask_feathered

    return mask_out.astype(np.uint8)


# ==================== ✅ 新增C：DSP 线条增强与双边滤波模块 ====================
# 原代码：inpainting 前不对原图做任何预处理
# 新代码：先对输入图像做双边滤波去噪 + Unsharp Mask 线条增强，保留细节

def dsp_enhance_image(image_np: np.ndarray,
                      bilateral_d: int = 9,
                      bilateral_sigma_color: float = 75,
                      bilateral_sigma_space: float = 75,
                      unsharp_strength: float = 0.5,
                      unsharp_sigma: int = 3) -> np.ndarray:
    """
    DSP 图像预处理：双边滤波去噪 + Unsharp Mask 线条增强
    Args:
        image_np              : RGB uint8 图像
        bilateral_d           : 双边滤波邻域直径
        bilateral_sigma_color : 双边滤波颜色空间 sigma
        bilateral_sigma_space : 双边滤波坐标空间 sigma
        unsharp_strength      : Unsharp Mask 强度（0=不增强，1=最强）
        unsharp_sigma         : Unsharp Mask 模糊核大小（奇数）
    Returns:
        预处理后的 RGB uint8 图像
    """
    # Step 1：双边滤波（保边去噪）
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    img_filtered = cv2.bilateralFilter(
        img_bgr, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)

    # Step 2：Unsharp Mask 线条增强
    ksize = unsharp_sigma if unsharp_sigma % 2 == 1 else unsharp_sigma + 1
    img_blurred = cv2.GaussianBlur(img_filtered, (ksize * 2 + 1, ksize * 2 + 1), unsharp_sigma)
    img_sharpened = cv2.addWeighted(
        img_filtered, 1.0 + unsharp_strength,
        img_blurred, -unsharp_strength,
        0
    )
    img_sharpened = np.clip(img_sharpened, 0, 255).astype(np.uint8)

    return cv2.cvtColor(img_sharpened, cv2.COLOR_BGR2RGB)


# ==================== ✅ 新增D：拉普拉斯金字塔融合模块 ====================
# 原代码：直接用 mask 将 inpainted 区域覆盖到原图（或不做融合直接保存 inpainted 输出）
# 新代码：使用拉普拉斯金字塔多尺度融合，消除边界接缝

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
    laplacian.append(gaussian[-1].astype(np.float32))  # 最顶层保留高斯
    return laplacian


def laplacian_pyramid_blend(orig_np: np.ndarray,
                             inpainted_np: np.ndarray,
                             mask_np: np.ndarray,
                             levels: int = 6) -> np.ndarray:
    """
    拉普拉斯金字塔融合
    Args:
        orig_np     : 原图 RGB uint8
        inpainted_np: inpainting 结果 RGB uint8（与 orig_np 同尺寸）
        mask_np     : uint8 mask（0=保留原图，255=使用 inpainted 区域）
        levels      : 金字塔层数
    Returns:
        融合后的 RGB uint8 图像
    """
    # 归一化 mask 到 [0,1]
    mask_f = mask_np.astype(np.float32) / 255.0
    if mask_f.ndim == 2:
        mask_f = np.stack([mask_f] * 3, axis=-1)

    # 构建各自金字塔
    lap_orig     = build_laplacian_pyramid(orig_np,      levels)
    lap_inpainted = build_laplacian_pyramid(inpainted_np, levels)
    gauss_mask   = build_gaussian_pyramid(mask_f,         levels)

    # 逐层融合
    blended_pyramid = []
    for lo, li, gm in zip(lap_orig, lap_inpainted, gauss_mask):
        blended = lo * (1 - gm) + li * gm
        blended_pyramid.append(blended)

    # 重建图像（从顶层往下 pyrUp）
    result = blended_pyramid[-1]
    for i in range(levels - 2, -1, -1):
        result = cv2.pyrUp(result, dstsize=(blended_pyramid[i].shape[1], blended_pyramid[i].shape[0]))
        result = result + blended_pyramid[i]

    return np.clip(result, 0, 255).astype(np.uint8)


# ==================== ✅ 新增B：ControlNet 管线加载函数 ====================
# 原代码：只加载 StableDiffusionXLInpaintPipeline
# 新代码：新增函数，可选加载带 ControlNet（Canny/Depth）约束的 inpainting 管线

def load_controlnet_pipe(sd_model_path: str,
                          controlnet_model_path: str,
                          device: str = "cuda") -> StableDiffusionXLControlNetInpaintPipeline:
    """
    加载带 ControlNet 约束的 SDXL Inpainting 管线
    Args:
        sd_model_path         : SDXL inpainting 基础模型路径
        controlnet_model_path : ControlNet 模型路径（如 canny/depth）
    Returns:
        StableDiffusionXLControlNetInpaintPipeline
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
                                   condition_type: str = "canny") -> Image.Image:
    """
    生成 ControlNet 条件图（Canny 边缘 / 深度图）
    Args:
        image_np       : RGB uint8 图像
        condition_type : "canny" 或 "depth"
    Returns:
        PIL.Image 条件图
    """
    if condition_type == "canny":
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, threshold1=100, threshold2=200)
        # 转为 RGB 三通道
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges_rgb)
    elif condition_type == "depth":
        # 简单灰度近似（生产环境建议用 MiDaS 或 ZoeDepth）
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        depth_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(depth_rgb)
    else:
        raise ValueError(f"不支持的 condition_type: {condition_type}")


# ==================== 评价指标类（保持不变） ====================

class Evaluator:
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

    def run(self, org_path, res_path, prompt):
        try:
            img_org = Image.open(org_path).convert('RGB').resize((512, 512))
            img_res = Image.open(res_path).convert('RGB').resize((512, 512))
            np_org = np.array(img_org)
            np_res = np.array(img_res)

            psnr_value = psnr_func(np_org, np_res, data_range=255)
            ssim_value = ssim_func(np_org, np_res, channel_axis=2, data_range=255)

            t_org = lpips.im2tensor(np_org).to(self.device)
            t_res = lpips.im2tensor(np_res).to(self.device)
            lpips_value = self.lpips_model(t_org, t_res).item()

            img_input = self.clip_preprocess(img_res).unsqueeze(0).to(self.device)
            text_input = self.tokenizer([prompt]).to(self.device)
            with torch.no_grad():
                img_feats = self.clip_model.encode_image(img_input)
                txt_feats = self.clip_model.encode_text(text_input)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
                clip_score = (img_feats @ txt_feats.T).item() * 100

            return {
                "PSNR": round(psnr_value, 2),
                "SSIM": round(ssim_value, 4),
                "LPIPS": round(lpips_value, 4),
                "CLIP_Score": round(clip_score, 2)
            }
        except Exception as e:
            print(f"    警告: 评价指标计算失败: {e}")
            return None


def load_test_configs(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


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
    inpaint_mode="first",
    nms_iou_threshold=0.5,
    inpaint_steps=50,
    inpaint_guidance_scale=7.5,
    inpaint_strength=0.99,
    inpaint_negative_prompt="blurry, bad quality, distorted, artifacts, ugly, low resolution",
    device="cuda",
    # ✅ 新增A：DSP mask 优化参数
    use_dsp_mask=True,
    dsp_morph_close_ksize=21,
    dsp_morph_open_ksize=7,
    dsp_feather_sigma=21,
    dsp_feather_strength=1.0,
    # ✅ 新增B：ControlNet 参数
    use_controlnet=False,
    controlnet_condition_type="canny",
    controlnet_scale=0.5,
    # ✅ 新增C：DSP 线条增强参数
    use_dsp_enhance=True,
    dsp_bilateral_d=9,
    dsp_bilateral_sigma=75,
    dsp_unsharp_strength=0.5,
    # ✅ 新增D：拉普拉斯金字塔融合参数
    use_lap_blend=True,
    lap_blend_levels=6,
):
    try:
        print(f"\n{'='*80}")
        print(f"测试: {description}")
        print(f"图片: {image_path}")
        print(f"检测提示词: {det_prompt}")
        print(f"修复提示词: {inpaint_prompt}")
        print(f"DSP Mask优化: {use_dsp_mask} | DSP线条增强: {use_dsp_enhance} | "
              f"ControlNet: {use_controlnet} | 拉普拉斯融合: {use_lap_blend}")
        print(f"{'='*80}\n")

        output_dir = os.path.join(output_base_dir, Path(image_path).parent.name, Path(image_path).stem)
        os.makedirs(output_dir, exist_ok=True)

        # [1/7] 加载图片
        print("  [1/7] 加载图片...")
        image_pil, image = load_image(image_path)
        raw_image_path = os.path.join(output_dir, "raw_image.jpg")
        image_pil.save(raw_image_path)

        # [2/7] Grounding DINO 检测
        print("  [2/7] 运行 Grounding DINO...")
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

        # [3/7] SAM 分割
        print("  [3/7] 运行 SAM 分割...")
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
        image_cv = cv2.imread(image_path)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

        # ✅ 新增C：DSP 线条增强与双边滤波（在 SAM 之后、inpainting 之前处理图像）
        # 原代码：直接使用 image_cv 送入后续流程
        # 新代码：可选地对 image_cv 做双边滤波去噪 + Unsharp Mask 线条增强
        if use_dsp_enhance:
            print("  [DSP-C] 线条增强与双边滤波...")
            image_cv_enhanced = dsp_enhance_image(
                image_cv,
                bilateral_d=dsp_bilateral_d,
                bilateral_sigma_color=dsp_bilateral_sigma,
                bilateral_sigma_space=dsp_bilateral_sigma,
                unsharp_strength=dsp_unsharp_strength,
            )
            # 保存增强后图像供调试
            Image.fromarray(image_cv_enhanced).save(
                os.path.join(output_dir, "dsp_enhanced.jpg"))
            print(f"  ✓ DSP增强完成，已保存 dsp_enhanced.jpg")
        else:
            image_cv_enhanced = image_cv  # 不增强时，直接使用原图

        predictor.set_image(image_cv)  # SAM 仍使用原始图像做分割（避免增强影响分割准确性）

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

        # [4/7] 保存分割结果
        print("  [4/7] 保存分割结果...")
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

        # [5/7] Mask 处理
        print("  [5/7] 处理 Mask...")

        if inpaint_mode == 'merge':
            raw_mask_np = torch.any(masks, dim=0)[0].cpu().numpy().astype(np.uint8) * 255
        else:
            raw_mask_np = masks[0][0].cpu().numpy().astype(np.uint8) * 255

        # ✅ 新增A：DSP Mask 优化（替换原来的简单膨胀+高斯模糊）
        # 原代码：
        #   kernel = np.ones((15, 15), np.uint8)
        #   merged_mask_np = cv2.dilate(merged_mask_np, kernel, iterations=1)
        #   merged_mask_np = cv2.GaussianBlur(merged_mask_np, (21, 21), 0)
        # 新代码：使用 dsp_optimize_mask 函数，增加形态学修补 + 可控羽化
        if use_dsp_mask:
            print("  [DSP-A] 执行 DSP Mask 优化（形态学修补 + 羽化）...")
            merged_mask_np = dsp_optimize_mask(
                raw_mask_np,
                morph_close_ksize=dsp_morph_close_ksize,
                morph_open_ksize=dsp_morph_open_ksize,
                dilate_ksize=15,
                dilate_iters=1,
                feather_sigma=dsp_feather_sigma,
                feather_strength=dsp_feather_strength,
            )
            # 保存优化后的 mask 供对比
            Image.fromarray(merged_mask_np).save(
                os.path.join(output_dir, "dsp_mask_optimized.jpg"))
            print(f"  ✓ DSP Mask 优化完成，已保存 dsp_mask_optimized.jpg")
        else:
            # 原始处理方式（兜底保留）
            kernel = np.ones((15, 15), np.uint8)
            merged_mask_np = cv2.dilate(raw_mask_np, kernel, iterations=1)
            merged_mask_np = cv2.GaussianBlur(merged_mask_np, (21, 21), 0)

        mask_pil = Image.fromarray(merged_mask_np)
        # ✅ 新增C：inpainting 输入使用 DSP 增强后的图像（而非原始图像）
        # 原代码：image_pil_for_inpaint = Image.fromarray(image_cv)
        # 新代码：根据开关选择增强图像或原图
        image_pil_for_inpaint = Image.fromarray(image_cv_enhanced)

        original_size = image_pil_for_inpaint.size
        image_padded, new_wh, offset = resize_with_padding(image_pil_for_inpaint, target_size=1024)
        mask_padded, _, _ = resize_with_padding(mask_pil.convert("RGB"), target_size=1024)
        mask_padded = mask_padded.convert("L")

        # [6/7] Inpainting（含 ControlNet 分支）
        print("  [6/7] 运行 Inpainting...")

        # ✅ 新增B：ControlNet 约束下的 SD Inpainting
        # 原代码：直接调用 pipe(prompt=..., image=..., mask_image=...)
        # 新代码：若 use_controlnet=True，额外生成条件图并通过 controlnet_conditioning_image 传入
        if use_controlnet:
            print(f"  [DSP-B] 生成 ControlNet 条件图（{controlnet_condition_type}）...")
            # 使用原始（未 padding）的增强图生成条件图
            condition_image = generate_controlnet_condition(
                np.array(image_pil_for_inpaint), condition_type=controlnet_condition_type)
            # 保存条件图供调试
            condition_image.save(os.path.join(output_dir, f"controlnet_condition_{controlnet_condition_type}.jpg"))
            # padding 条件图至与输入一致
            condition_padded, _, _ = resize_with_padding(condition_image, target_size=1024)

            inpainted_image = pipe(
                prompt=inpaint_prompt,
                negative_prompt=inpaint_negative_prompt,
                image=image_padded,
                mask_image=mask_padded,
                control_image=condition_padded,           # ← 新增B：ControlNet 条件图
                controlnet_conditioning_scale=controlnet_scale,  # ← 新增B：约束强度
                num_inference_steps=inpaint_steps,
                guidance_scale=inpaint_guidance_scale,
                strength=inpaint_strength,
            ).images[0]
            print(f"  ✓ ControlNet Inpainting 完成（scale={controlnet_scale}）")
        else:
            inpainted_image = pipe(
                prompt=inpaint_prompt,
                negative_prompt=inpaint_negative_prompt,
                image=image_padded,
                mask_image=mask_padded,
                num_inference_steps=inpaint_steps,
                guidance_scale=inpaint_guidance_scale,
                strength=inpaint_strength,
            ).images[0]

        # 还原 padding 到原始尺寸
        inpainted_image = restore_from_padding(inpainted_image, original_size, new_wh, offset, target_size=1024)
        inpainted_np = np.array(inpainted_image)

        # ✅ 新增D：拉普拉斯金字塔融合（替换直接覆盖合成）
        # 原代码：直接保存 inpainted_image（无显式融合，边界可能有接缝）
        # 新代码：使用拉普拉斯金字塔将 inpainted 区域与原图平滑融合
        if use_lap_blend:
            print(f"  [DSP-D] 拉普拉斯金字塔融合（levels={lap_blend_levels}）...")
            # 确保 mask 与原图同尺寸
            mask_for_blend = cv2.resize(
                merged_mask_np, (image_cv_enhanced.shape[1], image_cv_enhanced.shape[0]),
                interpolation=cv2.INTER_LINEAR)
            inpainted_resized = cv2.resize(
                inpainted_np, (image_cv_enhanced.shape[1], image_cv_enhanced.shape[0]),
                interpolation=cv2.INTER_LINEAR)

            blended_np = laplacian_pyramid_blend(
                orig_np=image_cv_enhanced,
                inpainted_np=inpainted_resized,
                mask_np=mask_for_blend,
                levels=lap_blend_levels,
            )
            final_image = Image.fromarray(blended_np)
            # 同时保存未融合的 inpainting 结果供对比
            inpainted_image.save(os.path.join(output_dir, "inpainting_before_blend.jpg"))
            print(f"  ✓ 拉普拉斯金字塔融合完成，已保存 inpainting_before_blend.jpg 用于对比")
        else:
            final_image = inpainted_image

        inpainting_path = os.path.join(output_dir, "grounded_sam_inpainting_output.jpg")
        final_image.save(inpainting_path)

        # [7/7] 评价指标
        metrics = None
        if evaluator:
            print("  [7/7] 计算评价指标...")
            metrics = evaluator.run(raw_image_path, inpainting_path, inpaint_prompt)
            if metrics:
                print(f"  ✓ PSNR: {metrics['PSNR']:.2f}, SSIM: {metrics['SSIM']:.4f}, "
                      f"LPIPS: {metrics['LPIPS']:.4f}, CLIP: {metrics['CLIP_Score']:.2f}")

        print(f"  ✓ 测试完成! 输出: {output_dir}")

        return {
            "status": "success",
            "image_path": image_path,
            "description": description,
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
        config = "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        grounded_checkpoint = "Grounded-Segment-Anything/weights/groundingdino_swint_ogc.pth"
        sam_checkpoint = "Grounded-Segment-Anything/weights/sam_vit_h_4b8939.pth"
        image_root = "Grounded-Segment-Anything/dataset"
        prompts_config = "Grounded-Segment-Anything/prompts_config.json"
        output_dir = "test_outputs"
        box_threshold = 0.3
        text_threshold = 0.25
        inpaint_mode = "first"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        test_subset = None
        enable_metrics = True
        sd_model_path = "/root/autodl-tmp/models/sdxl-inpainting"
        nms_iou_threshold = 0.5
        inpaint_steps = 50
        inpaint_guidance_scale = 7.5
        inpaint_strength = 0.99
        inpaint_negative_prompt = "blurry, bad quality, distorted, artifacts, ugly, low resolution"

        # ✅ 新增A：DSP Mask 优化开关及参数
        use_dsp_mask = True
        dsp_morph_close_ksize = 21   # 闭运算核：填补内部空洞
        dsp_morph_open_ksize  = 7    # 开运算核：去除碎点
        dsp_feather_sigma     = 21   # 羽化高斯 sigma
        dsp_feather_strength  = 1.0  # 羽化强度（0=硬边，1=完全羽化）

        # ✅ 新增B：ControlNet 开关及参数
        # 注意：use_controlnet=True 时，pipe 必须由 load_controlnet_pipe 加载
        use_controlnet           = False
        controlnet_model_path    = "/root/autodl-tmp/models/controlnet-canny-sdxl"
        controlnet_condition_type = "canny"   # "canny" 或 "depth"
        controlnet_scale          = 0.5       # ControlNet 约束强度 [0,1]

        # ✅ 新增C：DSP 线条增强开关及参数
        use_dsp_enhance       = True
        dsp_bilateral_d       = 9     # 双边滤波邻域直径
        dsp_bilateral_sigma   = 75    # 双边滤波 sigma
        dsp_unsharp_strength  = 0.5   # Unsharp Mask 强度

        # ✅ 新增D：拉普拉斯金字塔融合开关及参数
        use_lap_blend     = True
        lap_blend_levels  = 6   # 金字塔层数（越多，融合越平滑，但计算量越大）

    args = Args()

    print(f"加载提示词配置: {args.prompts_config}")
    test_configs = load_test_configs(args.prompts_config)
    print(f"✓ 加载了 {len(test_configs)} 个测试配置\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = os.path.join(args.output_dir, f"batch_test_{timestamp}")
    os.makedirs(output_base_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"开始批量测试")
    print(f"输出目录: {output_base_dir}")
    print(f"设备: {args.device}")
    print(f"计算评价指标: {'是' if args.enable_metrics else '否'}")
    print(f"DSP Mask优化: {args.use_dsp_mask} | DSP线条增强: {args.use_dsp_enhance}")
    print(f"ControlNet: {args.use_controlnet} | 拉普拉斯金字塔融合: {args.use_lap_blend}")
    print(f"{'='*80}\n")

    print("加载 Grounding DINO 模型...")
    model = load_model(args.config, args.grounded_checkpoint, device=args.device)
    print("✓ Grounding DINO 加载完成\n")

    # ✅ 新增B：根据 use_controlnet 选择加载哪种管线
    # 原代码：只加载 StableDiffusionXLInpaintPipeline
    # 新代码：use_controlnet=True 时加载 ControlNet 管线，否则加载普通管线
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
    total = len(test_configs)

    for idx, (rel_path, config) in enumerate(test_configs.items(), 1):
        print(f"\n进度: [{idx}/{total}]")

        image_path = os.path.join(args.image_root, rel_path)

        if not os.path.exists(image_path):
            print(f"警告: 图片不存在: {image_path}")
            results.append({
                "status": "error",
                "image_path": rel_path,
                "description": config["description"],
                "error": "文件不存在"
            })
            continue

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
            # ✅ 新增A 参数透传
            use_dsp_mask=args.use_dsp_mask,
            dsp_morph_close_ksize=args.dsp_morph_close_ksize,
            dsp_morph_open_ksize=args.dsp_morph_open_ksize,
            dsp_feather_sigma=args.dsp_feather_sigma,
            dsp_feather_strength=args.dsp_feather_strength,
            # ✅ 新增B 参数透传
            use_controlnet=args.use_controlnet,
            controlnet_condition_type=args.controlnet_condition_type,
            controlnet_scale=args.controlnet_scale,
            # ✅ 新增C 参数透传
            use_dsp_enhance=args.use_dsp_enhance,
            dsp_bilateral_d=args.dsp_bilateral_d,
            dsp_bilateral_sigma=args.dsp_bilateral_sigma,
            dsp_unsharp_strength=args.dsp_unsharp_strength,
            # ✅ 新增D 参数透传
            use_lap_blend=args.use_lap_blend,
            lap_blend_levels=args.lap_blend_levels,
        )
        results.append(result)

    # 汇总报告
    success_results = [r for r in results if r["status"] == "success"]
    report_path = os.path.join(output_base_dir, "test_report.json")
    report_data = {
        "timestamp": timestamp,
        "sd_model": args.sd_model_path,
        "inpaint_steps": args.inpaint_steps,
        "inpaint_guidance_scale": args.inpaint_guidance_scale,
        "inpaint_strength": args.inpaint_strength,
        "nms_iou_threshold": args.nms_iou_threshold,
        # ✅ 新增：在报告中记录四项新功能的配置
        "dsp_mask_enabled": args.use_dsp_mask,
        "dsp_enhance_enabled": args.use_dsp_enhance,
        "controlnet_enabled": args.use_controlnet,
        "lap_blend_enabled": args.use_lap_blend,
        "total_tests": total,
        "results": results,
        "summary": {
            "success": sum(1 for r in results if r["status"] == "success"),
            "warning": sum(1 for r in results if r["status"] == "warning"),
            "error": sum(1 for r in results if r["status"] == "error")
        }
    }

    if args.enable_metrics and success_results:
        metrics_list = [r["metrics"] for r in success_results if r.get("metrics")]
        if metrics_list:
            report_data["summary"]["average_metrics"] = {
                "PSNR": round(np.mean([m["PSNR"] for m in metrics_list]), 2),
                "SSIM": round(np.mean([m["SSIM"] for m in metrics_list]), 4),
                "LPIPS": round(np.mean([m["LPIPS"] for m in metrics_list]), 4),
                "CLIP_Score": round(np.mean([m["CLIP_Score"] for m in metrics_list]), 2)
            }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"批量测试完成!")
    print(f"{'='*80}")
    print(f"总测试数: {total}")
    print(f"  ✓ 成功: {report_data['summary']['success']}")
    print(f"  ⚠ 警告: {report_data['summary']['warning']}")
    print(f"  ✗ 失败: {report_data['summary']['error']}")

    if args.enable_metrics and 'average_metrics' in report_data['summary']:
        avg = report_data['summary']['average_metrics']
        print(f"\n平均评价指标:")
        print(f"  PSNR:       {avg['PSNR']:.2f} dB")
        print(f"  SSIM:       {avg['SSIM']:.4f}")
        print(f"  LPIPS:      {avg['LPIPS']:.4f}")
        print(f"  CLIP Score: {avg['CLIP_Score']:.2f}")

    print(f"\n详细报告: {report_path}")
    print(f"输出目录: {output_base_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()