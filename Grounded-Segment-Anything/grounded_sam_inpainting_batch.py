"""
批量测试 Grounded-SAM Inpainting
用于测试不同风格图片的分割和修复效果
"""

import os
import sys
import argparse
from pathlib import Path
import json
from datetime import datetime

# 导入原始脚本的函数（假设grounded_sam_inpainting_demo.py在同一目录下）
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
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有依赖已安装")
    sys.exit(1)


# 定义测试配置
TEST_CONFIGS = {
    # 美式漫画 - 卡通风格
    "AmericanComic/cartoon/1.jpg": {
        "det_prompt": "character",
        "inpaint_prompt": "cartoon character, vibrant colors, high quality",
        "description": "美式卡通角色测试"
    },
    "AmericanComic/cartoon/2.jpg": {
        "det_prompt": "character",
        "inpaint_prompt": "cartoon character, comic style, detailed",
        "description": "美式卡通角色测试2"
    },
    "AmericanComic/cartoon/3.jpg": {
        "det_prompt": "character",
        "inpaint_prompt": "cartoon character, bright colors, professional",
        "description": "美式卡通角色测试3"
    },
    "AmericanComic/cartoon/4.jpg": {
        "det_prompt": "character",
        "inpaint_prompt": "cartoon character, comic book style, detailed",
        "description": "美式卡通角色测试4"
    },
    "AmericanComic/cartoon/5.jpg": {
        "det_prompt": "character",
        "inpaint_prompt": "cartoon character, animated style, high quality",
        "description": "美式卡通角色测试5"
    },
    
    # 美式漫画 - 漫威风格
    "AmericanComic/Marvel/1.jpg": {
        "det_prompt": "superhero",
        "inpaint_prompt": "superhero, Marvel style, epic, detailed",
        "description": "漫威超级英雄测试"
    },
    "AmericanComic/Marvel/2.jpg": {
        "det_prompt": "superhero",
        "inpaint_prompt": "superhero, Marvel comics style, powerful",
        "description": "漫威超级英雄测试2"
    },
    "AmericanComic/Marvel/3.jpg": {
        "det_prompt": "superhero",
        "inpaint_prompt": "superhero, Marvel universe, high quality",
        "description": "漫威超级英雄测试3"
    },
    "AmericanComic/Marvel/4.jpg": {
        "det_prompt": "superhero",
        "inpaint_prompt": "superhero, Marvel style, cinematic",
        "description": "漫威超级英雄测试4"
    },
    "AmericanComic/Marvel/5.jpg": {
        "det_prompt": "superhero",
        "inpaint_prompt": "superhero, Marvel comics, detailed illustration",
        "description": "漫威超级英雄测试5"
    },
    
    # 中国风 - 古典
    "Chinese/classical/1.jpg": {
        "det_prompt": "person",
        "inpaint_prompt": "Chinese classical style, traditional, elegant, detailed",
        "description": "中国古典风格测试"
    },
    "Chinese/classical/2.jpg": {
        "det_prompt": "person",
        "inpaint_prompt": "Chinese classical painting style, traditional art, refined",
        "description": "中国古典风格测试2"
    },
    "Chinese/classical/3.jpg": {
        "det_prompt": "person",
        "inpaint_prompt": "Chinese classical art, traditional style, artistic",
        "description": "中国古典风格测试3"
    },
    "Chinese/classical/4.jpg": {
        "det_prompt": "person",
        "inpaint_prompt": "Chinese traditional painting, classical style, elegant",
        "description": "中国古典风格测试4"
    },
    "Chinese/classical/5.jpg": {
        "det_prompt": "person",
        "inpaint_prompt": "Chinese classical art, traditional aesthetics, detailed",
        "description": "中国古典风格测试5"
    },
    
    # 中国风 - 现代
    "Chinese/modern/1.jpg": {
        "det_prompt": "person",
        "inpaint_prompt": "modern Chinese style, contemporary art, high quality",
        "description": "中国现代风格测试"
    },
    "Chinese/modern/2.jpg": {
        "det_prompt": "person",
        "inpaint_prompt": "modern Chinese illustration, contemporary style, detailed",
        "description": "中国现代风格测试2"
    },
    "Chinese/modern/3.jpg": {
        "det_prompt": "person",
        "inpaint_prompt": "modern Chinese art, contemporary design, professional",
        "description": "中国现代风格测试3"
    },
    "Chinese/modern/4.jpg": {
        "det_prompt": "person",
        "inpaint_prompt": "modern Chinese style, contemporary illustration, high quality",
        "description": "中国现代风格测试4"
    },
    "Chinese/modern/5.jpg": {
        "det_prompt": "person",
        "inpaint_prompt": "modern Chinese art, contemporary aesthetics, detailed",
        "description": "中国现代风格测试5"
    },
    
    # 日本风格
    "Japanese/1.jpg": {
        "det_prompt": "character",
        "inpaint_prompt": "anime character, Japanese manga style, detailed",
        "description": "日本动漫风格测试"
    },
    "Japanese/2.jpg": {
        "det_prompt": "character",
        "inpaint_prompt": "anime style, Japanese illustration, high quality",
        "description": "日本动漫风格测试2"
    },
    "Japanese/3.jpg": {
        "det_prompt": "character",
        "inpaint_prompt": "manga character, Japanese anime style, detailed",
        "description": "日本动漫风格测试3"
    },
    "Japanese/4.jpg": {
        "det_prompt": "character",
        "inpaint_prompt": "anime style, Japanese manga art, professional",
        "description": "日本动漫风格测试4"
    },
    "Japanese/5.jpg": {
        "det_prompt": "character",
        "inpaint_prompt": "manga style, Japanese anime illustration, high quality",
        "description": "日本动漫风格测试5"
    },
}


def run_single_test(
    image_path,
    det_prompt,
    inpaint_prompt,
    description,
    model,
    sam_checkpoint,
    output_base_dir,
    box_threshold=0.3,
    text_threshold=0.25,
    inpaint_mode="first",
    device="cpu"
):
    """
    运行单个图片的测试
    
    Args:
        image_path: 输入图片路径
        det_prompt: 检测提示词
        inpaint_prompt: 修复提示词
        description: 测试描述
        model: 加载的 Grounding DINO 模型
        sam_checkpoint: SAM 模型检查点路径
        output_base_dir: 输出基础目录
        box_threshold: 边界框阈值
        text_threshold: 文本阈值
        inpaint_mode: 修复模式
        device: 运行设备
    
    Returns:
        dict: 测试结果
    """
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
        print("加载图片...")
        image_pil, image = load_image(image_path)
        
        # 保存原始图片
        image_pil.save(os.path.join(output_dir, "raw_image.jpg"))
        
        # 运行 Grounding DINO
        print("运行 Grounding DINO 模型...")
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, det_prompt, box_threshold, text_threshold, device=device
        )
        
        print(f"检测到 {len(boxes_filt)} 个目标: {pred_phrases}")
        
        if len(boxes_filt) == 0:
            print("警告: 未检测到任何目标!")
            return {
                "status": "warning",
                "message": "未检测到任何目标",
                "image_path": image_path,
                "description": description
            }
        
        # 初始化 SAM
        print("初始化 SAM...")
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
        print("生成分割 mask...")
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(device),
            multimask_output=False,
        )
        
        # 绘制输出图片
        print("绘制分割结果...")
        plt.figure(figsize=(10, 10))
        plt.imshow(image_cv)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box.numpy(), plt.gca(), label)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "grounded_sam_output.jpg"), bbox_inches="tight", dpi=150)
        plt.close()
        
        # 修复（Inpainting）
        print("运行 Inpainting...")
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
        inpainted_image.save(os.path.join(output_dir, "grounded_sam_inpainting_output.jpg"))
        
        print(f"✓ 测试完成! 输出保存到: {output_dir}")
        
        return {
            "status": "success",
            "image_path": image_path,
            "description": description,
            "output_dir": output_dir,
            "detected_objects": len(boxes_filt),
            "detected_labels": pred_phrases
        }
        
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "image_path": image_path,
            "description": description,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser("批量测试 Grounded-SAM Inpainting", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="Grounding DINO 配置文件路径")
    parser.add_argument("--grounded_checkpoint", type=str, required=True, help="Grounding DINO 检查点路径")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="SAM 检查点路径")
    parser.add_argument("--image_root", type=str, required=True, help="图片根目录")
    parser.add_argument("--output_dir", type=str, default="test_outputs", help="输出目录")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="边界框阈值")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="文本阈值")
    parser.add_argument("--inpaint_mode", type=str, default="first", help="修复模式")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="运行设备")
    parser.add_argument("--test_subset", type=str, nargs="+", help="仅测试指定的图片子集（例如: AmericanComic/cartoon/1.jpg）")
    
    args = parser.parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = os.path.join(args.output_dir, f"batch_test_{timestamp}")
    os.makedirs(output_base_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"开始批量测试")
    print(f"输出目录: {output_base_dir}")
    print(f"设备: {args.device}")
    print(f"{'='*80}\n")
    
    # 加载模型
    print("加载 Grounding DINO 模型...")
    model = load_model(args.config, args.grounded_checkpoint, device=args.device)
    print("✓ 模型加载完成\n")
    
    # 确定要测试的图片
    if args.test_subset:
        test_configs = {k: v for k, v in TEST_CONFIGS.items() if k in args.test_subset}
    else:
        test_configs = TEST_CONFIGS
    
    # 运行测试
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
            output_base_dir=output_base_dir,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            inpaint_mode=args.inpaint_mode,
            device=args.device
        )
        
        results.append(result)
    
    # 保存测试报告
    report_path = os.path.join(output_base_dir, "test_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "total_tests": total,
            "results": results,
            "summary": {
                "success": sum(1 for r in results if r["status"] == "success"),
                "warning": sum(1 for r in results if r["status"] == "warning"),
                "error": sum(1 for r in results if r["status"] == "error")
            }
        }, f, indent=2, ensure_ascii=False)
    
    # 打印总结
    print(f"\n{'='*80}")
    print(f"测试完成!")
    print(f"{'='*80}")
    print(f"总测试数: {total}")
    print(f"成功: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"警告: {sum(1 for r in results if r['status'] == 'warning')}")
    print(f"失败: {sum(1 for r in results if r['status'] == 'error')}")
    print(f"\n详细报告: {report_path}")
    print(f"输出目录: {output_base_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()