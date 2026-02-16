"""
单张图片快速测试脚本（带评价指标）
用于快速验证 Grounded-SAM Inpainting 效果并计算评价指标

评价指标：
- PSNR (Peak Signal-to-Noise Ratio): 峰值信噪比，越高越好
- SSIM (Structural Similarity Index): 结构相似性，越高越好（0-1）
- LPIPS (Learned Perceptual Image Patch Similarity): 感知相似度，越低越好
- CLIP Score: 文本-图像相似度，越高越好
"""

import os
import sys
from pathlib import Path
import json

try:
    from grounded_sam_inpainting_demo import (
        load_image, 
        load_model, 
        get_grounding_output,
        show_mask,
        show_box
    )
    import torch
    import cv2
    from PIL import Image
    from segment_anything import SamPredictor, build_sam
    from diffusers import StableDiffusionInpaintPipeline
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio as psnr_func
    from skimage.metrics import structural_similarity as ssim_func
    import lpips
    import open_clip
except ImportError as e:
    print(f"导入错误: {e}")
    print("\n请确保所有依赖已安装:")
    print("  pip install torch torchvision")
    print("  pip install opencv-python pillow matplotlib numpy")
    print("  pip install scikit-image")
    print("  pip install lpips")
    print("  pip install open_clip_torch")
    sys.exit(1)


# ==================== 配置区域 ====================
# 在这里直接修改配置，无需命令行参数

CONFIG = {
    # 模型路径配置
    "grounding_dino_config": "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",  # 修改为你的配置文件路径
    "grounded_checkpoint": "Grounded-Segment-Anything/weights/groundingdino_swint_ogc.pth",  # 修改为你的 Grounding DINO 检查点路径
    "sam_checkpoint": "Grounded-Segment-Anything/weights/sam_vit_h_4b8939.pth",              # 修改为你的 SAM 检查点路径
    
    # 测试图片配置
    "image_path": "Grounded-Segment-Anything/dataset/AmericanComic/cartoon/1.jpg",  # 修改为你要测试的图片路径
    
    # 提示词配置
    "det_prompt": "fan",  # 检测提示词（用于 Grounding DINO）
    "inpaint_prompt": "cartoon character, vibrant colors, high quality, detailed, bag",  # 修复提示词（用于 Stable Diffusion）
    
    # 输出配置
    "output_dir": "single_test_output",
    
    # 模型参数
    "box_threshold": 0.3,      # 边界框阈值（0-1），越低检测越多目标
    "text_threshold": 0.25,    # 文本阈值（0-1）
    "device": "cuda",          # 设备选择: "cuda" 或 "cpu"
    
    # 评价指标配置
    "enable_metrics": True,    # 是否计算评价指标
}

# ==================== 评价指标类 ====================

class Evaluator:
    """评价指标计算类"""
    
    def __init__(self, device):
        """
        初始化评价器
        
        Args:
            device: 计算设备 ("cuda" 或 "cpu")
        """
        self.device = device
        
        print("初始化评价指标模型...")
        # 1. LPIPS 模型（感知相似度）
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        
        # 2. CLIP 模型（文本-图像相似度）
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', 
            pretrained='laion2b_s34b_b79k'
        )
        self.clip_model = self.clip_model.to(device)
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        print("✓ 评价指标模型加载完成\n")
    
    def run(self, org_path, res_path, prompt):
        """
        计算所有评价指标
        
        Args:
            org_path: 原始图片路径
            res_path: 修复结果图片路径
            prompt: 修复提示词（用于 CLIP Score）
        
        Returns:
            dict: 包含所有评价指标的字典
        """
        # 加载图片并统一尺寸
        img_org = Image.open(org_path).convert('RGB').resize((512, 512))
        img_res = Image.open(res_path).convert('RGB').resize((512, 512))
        
        # 转换为 numpy 数组
        np_org = np.array(img_org)
        np_res = np.array(img_res)
        
        # 1. PSNR (峰值信噪比)
        # 范围: 通常 20-50 dB，越高越好
        # 含义: 衡量图像质量，值越大表示失真越小
        psnr_value = psnr_func(np_org, np_res, data_range=255)
        
        # 2. SSIM (结构相似性)
        # 范围: 0-1，越接近 1 越好
        # 含义: 衡量两张图片的结构相似性
        ssim_value = ssim_func(np_org, np_res, channel_axis=2, data_range=255)
        
        # 3. LPIPS (感知相似度)
        # 范围: 0-1+，越小越好
        # 含义: 基于深度学习的感知相似度，更符合人类视觉
        t_org = lpips.im2tensor(np_org).to(self.device)
        t_res = lpips.im2tensor(np_res).to(self.device)
        lpips_value = self.lpips_model(t_org, t_res).item()
        
        # 4. CLIP Score (文本-图像相似度)
        # 范围: 通常 0-100，越高越好
        # 含义: 衡量图像与文本提示词的匹配程度
        img_input = self.clip_preprocess(img_res).unsqueeze(0).to(self.device)
        text_input = self.tokenizer([prompt]).to(self.device)
        
        with torch.no_grad():
            img_feats = self.clip_model.encode_image(img_input)
            txt_feats = self.clip_model.encode_text(text_input)
            # 归一化特征
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            # 计算相似度（余弦相似度）
            clip_score = (img_feats @ txt_feats.T).item() * 100  # 乘以100使其更易读
        
        return {
            "PSNR": round(psnr_value, 2),
            "SSIM": round(ssim_value, 4),
            "LPIPS": round(lpips_value, 4),
            "CLIP_Score": round(clip_score, 2)
        }


# ==================== 主函数 ====================

def main():
    """主测试函数"""
    
    # 从配置中读取参数
    config = CONFIG
    
    # 创建输出目录
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # 打印配置信息
    print(f"\n{'='*80}")
    print(f"单张图片快速测试（带评价指标）")
    print(f"{'='*80}")
    print(f"图片路径: {config['image_path']}")
    print(f"检测提示词: {config['det_prompt']}")
    print(f"修复提示词: {config['inpaint_prompt']}")
    print(f"输出目录: {config['output_dir']}")
    print(f"设备: {config['device']}")
    print(f"计算评价指标: {config['enable_metrics']}")
    print(f"{'='*80}\n")
    
    # 检查图片是否存在
    if not os.path.exists(config["image_path"]):
        print(f"错误: 图片不存在: {config['image_path']}")
        print("请修改脚本中的 CONFIG['image_path'] 配置")
        sys.exit(1)
    
    # 检查设备
    device = config["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA 不可用，切换到 CPU")
        device = "cpu"
    
    # ==================== 步骤 1: 加载模型 ====================
    print("步骤 1/7: 加载 Grounding DINO 模型...")
    try:
        model = load_model(
            config["grounding_dino_config"], 
            config["grounded_checkpoint"], 
            device=device
        )
        print("✓ Grounding DINO 模型加载完成\n")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        print("请检查配置文件和检查点路径是否正确")
        sys.exit(1)
    
    # ==================== 步骤 2: 加载图片 ====================
    print("步骤 2/7: 加载图片...")
    image_pil, image = load_image(config["image_path"])
    raw_image_path = os.path.join(config["output_dir"], "raw_image.jpg")
    image_pil.save(raw_image_path)
    print("✓ 图片加载完成\n")
    
    # ==================== 步骤 3: 运行检测 ====================
    print("步骤 3/7: 运行 Grounding DINO 目标检测...")
    boxes_filt, pred_phrases = get_grounding_output(
        model, 
        image, 
        config["det_prompt"], 
        config["box_threshold"], 
        config["text_threshold"], 
        device=device
    )
    print(f"✓ 检测到 {len(boxes_filt)} 个目标")
    if len(boxes_filt) > 0:
        print(f"  检测结果: {pred_phrases}\n")
    else:
        print("  未检测到任何目标!\n")
        print("建议:")
        print("  - 降低 CONFIG['box_threshold'] 值（例如: 0.2）")
        print("  - 修改 CONFIG['det_prompt'] 为更通用的词（例如: 'object'）")
        sys.exit(0)
    
    # ==================== 步骤 4: 运行分割 ====================
    print("步骤 4/7: 运行 SAM 分割...")
    predictor = SamPredictor(build_sam(checkpoint=config["sam_checkpoint"]).to(device))
    image_cv = cv2.imread(config["image_path"])
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_cv)
    
    # 转换边界框坐标
    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    
    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes_filt, 
        image_cv.shape[:2]
    ).to(device)
    
    # 生成分割 mask
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )
    print("✓ 分割完成\n")
    
    # ==================== 步骤 5: 保存分割结果 ====================
    print("步骤 5/7: 保存分割可视化结果...")
    plt.figure(figsize=(10, 10))
    plt.imshow(image_cv)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)
    plt.axis('off')
    segmentation_output_path = os.path.join(config["output_dir"], "grounded_sam_output.jpg")
    plt.savefig(segmentation_output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print("✓ 分割结果已保存\n")
    
    # ==================== 步骤 6: 运行修复 ====================
    print("步骤 6/7: 运行 Stable Diffusion Inpainting...")
    mask = masks[0][0].cpu().numpy()
    mask_pil = Image.fromarray(mask)
    image_pil_resized = Image.fromarray(image_cv)
    
    # 加载 Inpainting 管道
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # 调整尺寸
    image_pil_resized = image_pil_resized.resize((512, 512))
    mask_pil = mask_pil.resize((512, 512))
    
    # 生成修复图片
    inpainted_image = pipe(
        prompt=config["inpaint_prompt"], 
        image=image_pil_resized, 
        mask_image=mask_pil
    ).images[0]
    inpainted_image = inpainted_image.resize(size)
    inpainting_output_path = os.path.join(config["output_dir"], "grounded_sam_inpainting_output.jpg")
    inpainted_image.save(inpainting_output_path)
    print("✓ 修复完成\n")
    
    # ==================== 步骤 7: 计算评价指标 ====================
    metrics = None
    if config["enable_metrics"]:
        print("步骤 7/7: 计算评价指标...")
        try:
            evaluator = Evaluator(device)
            metrics = evaluator.run(
                raw_image_path, 
                inpainting_output_path, 
                config["inpaint_prompt"]
            )
            print("✓ 评价指标计算完成\n")
        except Exception as e:
            print(f"✗ 评价指标计算失败: {e}")
            print("继续执行，但无评价指标结果\n")
    else:
        print("步骤 7/7: 跳过评价指标计算（已禁用）\n")
    
    # ==================== 输出结果 ====================
    print(f"{'='*80}")
    print(f"测试完成!")
    print(f"{'='*80}")
    print(f"\n输出文件:")
    print(f"  ├─ 原始图片: {raw_image_path}")
    print(f"  ├─ 分割结果: {segmentation_output_path}")
    print(f"  └─ 修复结果: {inpainting_output_path}")
    
    if metrics:
        print(f"\n评价指标:")
        print(f"  ├─ PSNR (峰值信噪比):      {metrics['PSNR']:.2f} dB  {'(越高越好, 通常 20-50)' if metrics['PSNR'] < 100 else ''}")
        print(f"  ├─ SSIM (结构相似性):      {metrics['SSIM']:.4f}     (越高越好, 范围 0-1)")
        print(f"  ├─ LPIPS (感知相似度):     {metrics['LPIPS']:.4f}     (越低越好, 范围 0-1+)")
        print(f"  └─ CLIP Score (文本匹配):  {metrics['CLIP_Score']:.2f}      (越高越好, 通常 0-100)")
        
        # 评价指标解读
        print(f"\n指标解读:")
        if metrics['SSIM'] >= 0.9:
            print(f"  • 结构保持: 优秀 (SSIM ≥ 0.9)")
        elif metrics['SSIM'] >= 0.8:
            print(f"  • 结构保持: 良好 (SSIM ≥ 0.8)")
        else:
            print(f"  • 结构保持: 一般 (SSIM < 0.8)")
        
        if metrics['LPIPS'] <= 0.2:
            print(f"  • 感知质量: 优秀 (LPIPS ≤ 0.2)")
        elif metrics['LPIPS'] <= 0.4:
            print(f"  • 感知质量: 良好 (LPIPS ≤ 0.4)")
        else:
            print(f"  • 感知质量: 一般 (LPIPS > 0.4)")
        
        if metrics['CLIP_Score'] >= 30:
            print(f"  • 文本匹配: 优秀 (CLIP Score ≥ 30)")
        elif metrics['CLIP_Score'] >= 20:
            print(f"  • 文本匹配: 良好 (CLIP Score ≥ 20)")
        else:
            print(f"  • 文本匹配: 一般 (CLIP Score < 20)")
        
        # 保存指标到 JSON 文件
        metrics_path = os.path.join(config["output_dir"], "metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump({
                "image_path": config["image_path"],
                "det_prompt": config["det_prompt"],
                "inpaint_prompt": config["inpaint_prompt"],
                "detected_objects": len(boxes_filt),
                "detected_labels": pred_phrases,
                "metrics": metrics
            }, f, indent=2, ensure_ascii=False)
        print(f"\n  指标已保存到: {metrics_path}")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()