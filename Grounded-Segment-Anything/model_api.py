import torch
import numpy as np
import os
from PIL import Image

from grounded_sam_inpainting_v5 import (
    load_model,
    run_single_test,
    load_image,
    get_grounding_output,
    filter_boxes_nms,
)

from segment_anything import SamPredictor, build_sam
from diffusers import StableDiffusionXLInpaintPipeline


class InpaintModel:
    def __init__(self):
        print("🚀 初始化模型中...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # GroundingDINO
        self.model = load_model(
            "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "Grounded-Segment-Anything/weights/groundingdino_swint_ogc.pth",
            device=self.device
        )

        # SAM
        self.sam_checkpoint = "Grounded-Segment-Anything/weights/sam_vit_h_4b8939.pth"

        # SDXL（可选，如果太卡可以先注释）
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "/root/autodl-tmp/models/sdxl-inpainting",
            torch_dtype=torch.float16,
            local_files_only=True
        ).to(self.device)

        print("✅ 模型加载完成")

    # ===== Step1: 生成 Mask =====
    def get_mask(self, image_path, det_prompt):
        print("🟡 开始检测 + 分割")

        image_pil, image = load_image(image_path)

        boxes_filt, pred_phrases, scores = get_grounding_output(
            self.model, image, det_prompt, 0.3, 0.25, device=self.device
        )

        boxes_filt, pred_phrases, scores = filter_boxes_nms(
            boxes_filt, pred_phrases, scores, iou_threshold=0.5
        )

        if len(boxes_filt) == 0:
            print("⚠️ 没检测到目标")
            return None

        predictor = SamPredictor(build_sam(checkpoint=self.sam_checkpoint).to(self.device))

        image_np = np.array(image_pil)
        predictor.set_image(image_np)

        H, W = image_np.shape[:2]
        boxes_filt = boxes_filt.cpu()

        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        transformed_boxes = predictor.transform.apply_boxes_torch(
            boxes_filt, image_np.shape[:2]
        ).to(self.device)

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        mask = torch.any(masks, dim=0)[0].cpu().numpy().astype(np.uint8) * 255

        print("✅ Mask生成完成")

        return Image.fromarray(mask)

    # ===== Step2: 修复 =====
    def infer(self, image_path, det_prompt, inpaint_prompt, task_type):
        print("🔵 开始图像修复")

        result = run_single_test(
            image_path=image_path,
            det_prompt=det_prompt,
            inpaint_prompt=inpaint_prompt,
            description="gradio_demo",
            model=self.model,
            sam_checkpoint=self.sam_checkpoint,
            pipe=self.pipe,
            output_base_dir="gradio_outputs",
            evaluator=None,
            device=self.device,
            task_type=task_type,

            use_lama=True,
            use_zits=True,
            use_mat=True,
            use_two_stage=True,
            use_dsp_mask=True,
        )

        print("✅ 修复完成")

        return result