"""
批量测试 Grounded-SAM Inpainting（支持评价指标）
基于 test_grounded_sam_with_config.py，增加了评价指标计算功能
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

# 导入原始脚本的函数
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
    print("请确保所有依赖已安装")
    sys.exit(1)


# ==================== 评价指标类 ====================

class Evaluator:
    """评价指标计算类"""
    
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
        """计算所有评价指标"""
        try:
            img_org = Image.open(org_path).convert('RGB').resize((512, 512))
            img_res = Image.open(res_path).convert('RGB').resize((512, 512))
            
            np_org = np.array(img_org)
            np_res = np.array(img_res)
            
            # PSNR & SSIM
            psnr_value = psnr_func(np_org, np_res, data_range=255)
            ssim_value = ssim_func(np_org, np_res, channel_axis=2, data_range=255)
            
            # LPIPS
            t_org = lpips.im2tensor(np_org).to(self.device)
            t_res = lpips.im2tensor(np_res).to(self.device)
            lpips_value = self.lpips_model(t_org, t_res).item()
            
            # CLIP Score
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
    """从 JSON 文件加载测试配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_single_test(
    image_path,
    det_prompt,
    inpaint_prompt,
    description,
    model,
    sam_checkpoint,
    output_base_dir,
    evaluator=None,
    box_threshold=0.3,
    text_threshold=0.25,
    inpaint_mode="first",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """运行单个图片的测试（带评价指标）"""
    try:
        print(f"\n{'='*80}")
        print(f"测试: {description}")
        print(f"图片: {image_path}")
        print(f"检测提示词: {det_prompt}")
        print(f"修复提示词: {inpaint_prompt}")
        print(f"{'='*80}\n")
        
        # 创建输出目录
        output_dir = os.path.join(output_base_dir, Path(image_path).parent.name, Path(image_path).stem)
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载图片
        print("  [1/6] 加载图片...")
        image_pil, image = load_image(image_path)
        raw_image_path = os.path.join(output_dir, "raw_image.jpg")
        image_pil.save(raw_image_path)
        
        # 运行 Grounding DINO
        print("  [2/6] 运行 Grounding DINO...")
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, det_prompt, box_threshold, text_threshold, device=device
        )
        
        print(f"  ✓ 检测到 {len(boxes_filt)} 个目标: {pred_phrases}")
        
        if len(boxes_filt) == 0:
            print("  ✗ 警告: 未检测到任何目标!")
            return {
                "status": "warning",
                "message": "未检测到任何目标",
                "image_path": image_path,
                "description": description
            }
        
        # 初始化 SAM
        print("  [3/6] 运行 SAM 分割...")
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
        image_cv = cv2.imread(image_path)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_cv)
        
        # 转换边界框
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        
        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_cv.shape[:2]).to(device)
        
        # 生成 mask
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(device),
            multimask_output=False,
        )
        
        # 绘制输出图片
        print("  [4/6] 保存分割结果...")
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
        
        # 修复（Inpainting）
        print("  [5/6] 运行 Inpainting...")
        if inpaint_mode == 'merge':
            masks = torch.sum(masks, dim=0).unsqueeze(0)
            masks = torch.where(masks > 0, True, False)
        
        mask = masks[0][0].cpu().numpy()
        mask_pil = Image.fromarray(mask)
        image_pil_resized = Image.fromarray(image_cv)
        
        # 加载 Inpainting 管道
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        
        image_pil_resized = image_pil_resized.resize((512, 512))
        mask_pil = mask_pil.resize((512, 512))
        
        # 生成修复图片
        inpainted_image = pipe(prompt=inpaint_prompt, image=image_pil_resized, mask_image=mask_pil).images[0]
        inpainted_image = inpainted_image.resize(size)
        inpainting_path = os.path.join(output_dir, "grounded_sam_inpainting_output.jpg")
        inpainted_image.save(inpainting_path)
        
        # 计算评价指标
        metrics = None
        if evaluator:
            print("  [6/6] 计算评价指标...")
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
    parser = argparse.ArgumentParser("批量测试 Grounded-SAM Inpainting（带评价指标）", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--grounded_checkpoint", type=str, required=True, help="Grounded-Segment-Anything/weights/groundingdino_swint_ogc.pth")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="Grounded-Segment-Anything/weights/sam_vit_h_4b8939.pth")
    parser.add_argument("--image_root", type=str, required=True, help="Grounded-Segment-Anything/dataset")
    parser.add_argument("--prompts_config", type=str, default="prompts_config.json", help="Grounded-Segment-Anything/prompt_config.json")
    parser.add_argument("--output_dir", type=str, default="test_outputs", help="Grounded-Segment-Anything/output_dir")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="边界框阈值")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="文本阈值")
    parser.add_argument("--inpaint_mode", type=str, default="first", help="修复模式")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="运行设备")
    parser.add_argument("--test_subset", type=str, nargs="+", help="仅测试指定的图片子集")
    parser.add_argument("--enable_metrics", action="store_true", help="计算评价指标（会显著增加运行时间）")
    
    args = parser.parse_args()
    
    # 加载提示词配置
    print(f"加载提示词配置: {args.prompts_config}")
    test_configs = load_test_configs(args.prompts_config)
    print(f"✓ 加载了 {len(test_configs)} 个测试配置\n")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = os.path.join(args.output_dir, f"batch_test_{timestamp}")
    os.makedirs(output_base_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"开始批量测试")
    print(f"输出目录: {output_base_dir}")
    print(f"设备: {args.device}")
    print(f"计算评价指标: {'是' if args.enable_metrics else '否'}")
    print(f"{'='*80}\n")
    
    # 加载模型
    print("加载 Grounding DINO 模型...")
    model = load_model(args.config, args.grounded_checkpoint, device=args.device)
    print("✓ Grounding DINO 模型加载完成\n")
    
    # 初始化评价器（如果启用）
    evaluator = None
    if args.enable_metrics:
        evaluator = Evaluator(args.device)
    
    # 确定要测试的图片
    if args.test_subset:
        test_configs = {k: v for k, v in test_configs.items() if k in args.test_subset}
    
    # 运行测试
    results = []
    total = len(test_configs)
    
    for idx, (rel_path, config) in enumerate(test_configs.items(), 1):
        print(f"\n{'='*80}")
        print(f"进度: [{idx}/{total}]")
        print(f"{'='*80}")
        
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
            output_base_dir=output_base_dir,
            evaluator=evaluator,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            inpaint_mode=args.inpaint_mode,
            device=args.device
        )
        
        results.append(result)
    
    # 计算汇总统计
    success_results = [r for r in results if r["status"] == "success"]
    
    # 保存测试报告
    report_path = os.path.join(output_base_dir, "test_report.json")
    report_data = {
        "timestamp": timestamp,
        "total_tests": total,
        "results": results,
        "summary": {
            "success": sum(1 for r in results if r["status"] == "success"),
            "warning": sum(1 for r in results if r["status"] == "warning"),
            "error": sum(1 for r in results if r["status"] == "error")
        }
    }
    
    # 如果启用了评价指标，计算平均值
    if args.enable_metrics and success_results:
        metrics_list = [r["metrics"] for r in success_results if r.get("metrics")]
        if metrics_list:
            avg_metrics = {
                "PSNR": round(np.mean([m["PSNR"] for m in metrics_list]), 2),
                "SSIM": round(np.mean([m["SSIM"] for m in metrics_list]), 4),
                "LPIPS": round(np.mean([m["LPIPS"] for m in metrics_list]), 4),
                "CLIP_Score": round(np.mean([m["CLIP_Score"] for m in metrics_list]), 2)
            }
            report_data["summary"]["average_metrics"] = avg_metrics
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    # 打印总结
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