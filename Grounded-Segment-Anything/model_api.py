# import torch
# import numpy as np
# import os
# from PIL import Image

# from grounded_sam_inpainting_v5 import (
#     load_model,
#     run_single_test,
#     load_image,
#     get_grounding_output,
#     filter_boxes_nms,
# )

# from segment_anything import SamPredictor, build_sam
# from diffusers import StableDiffusionXLInpaintPipeline


# class InpaintModel:
#     def __init__(self):
#         print("🚀 初始化模型中...")

#         self.device = "cuda" if torch.cuda.is_available() else "cpu"

#         # GroundingDINO
#         self.model = load_model(
#             "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
#             "Grounded-Segment-Anything/weights/groundingdino_swint_ogc.pth",
#             device=self.device
#         )

#         # SAM
#         self.sam_checkpoint = "Grounded-Segment-Anything/weights/sam_vit_h_4b8939.pth"

#         # SDXL（可选，如果太卡可以先注释）
#         self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
#             "/root/autodl-tmp/models/sdxl-inpainting",
#             torch_dtype=torch.float16,
#             local_files_only=True
#         ).to(self.device)

#         print("✅ 模型加载完成")

#     # ===== Step1: 生成 Mask =====
#     def get_mask(self, image_path, det_prompt):
#         print("🟡 开始检测 + 分割")

#         image_pil, image = load_image(image_path)

#         boxes_filt, pred_phrases, scores = get_grounding_output(
#             self.model, image, det_prompt, 0.3, 0.25, device=self.device
#         )

#         boxes_filt, pred_phrases, scores = filter_boxes_nms(
#             boxes_filt, pred_phrases, scores, iou_threshold=0.5
#         )

#         if len(boxes_filt) == 0:
#             print("⚠️ 没检测到目标")
#             return None

#         predictor = SamPredictor(build_sam(checkpoint=self.sam_checkpoint).to(self.device))

#         image_np = np.array(image_pil)
#         predictor.set_image(image_np)

#         H, W = image_pil.size[1], image_pil.size[0]
#         boxes_filt = boxes_filt.cpu()

#         for i in range(boxes_filt.size(0)):
#             boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
#             boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
#             boxes_filt[i][2:] += boxes_filt[i][:2]

#         transformed_boxes = predictor.transform.apply_boxes_torch(
#             boxes_filt, image_np.shape[:2]
#         ).to(self.device)

#         masks, _, _ = predictor.predict_torch(
#             point_coords=None,
#             point_labels=None,
#             boxes=transformed_boxes,
#             multimask_output=False,
#         )

#         mask = torch.any(masks, dim=0)[0].cpu().numpy().astype(np.uint8) * 255

#         print("✅ Mask生成完成")

#         return Image.fromarray(mask)

#     # ===== Step2: 修复 =====
#     def infer(self, image_path, det_prompt, inpaint_prompt, task_type):
#         print("🔵 开始图像修复")

#         result = run_single_test(
#             image_path=image_path,
#             det_prompt=det_prompt,
#             inpaint_prompt=inpaint_prompt,
#             description="gradio_demo",
#             model=self.model,
#             sam_checkpoint=self.sam_checkpoint,
#             pipe=self.pipe,
#             output_base_dir="gradio_outputs",
#             evaluator=None,
#             device=self.device,
#             task_type=task_type,

#             use_lama=True,
#             use_zits=True,
#             use_mat=True,
#             use_two_stage=True,
#             use_dsp_mask=True,

#             mask_erode_ksize=9,
#             mask_erode_iters=1,
#             mask_smooth_sigma=15,
#         )


#         print("✅ 修复完成")

#         return result

import torch
import numpy as np
import os
from PIL import Image, ImageFilter

from grounded_sam_inpainting_v4_final import (
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

        # SDXL
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

        predictor = SamPredictor(
            build_sam(checkpoint=self.sam_checkpoint).to(self.device)
        )
        image_np = np.array(image_pil)
        predictor.set_image(image_np)

        H, W = image_pil.size[1], image_pil.size[0]
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
        print("✅ Mask 生成完成")
        return Image.fromarray(mask)

    # ===== Step2: 修复 =====
    def infer(self, image_path, det_prompt, inpaint_prompt, task_type,
              mask_override: str | None = None):
        """
        mask_override: 可选，自定义 mask 图片路径（PNG，L 模式）。
                       传入时跳过检测 + SAM，直接使用该 mask 调用 SDXL。
                       为 None 时走原有 run_single_test 全流程。
        """

        # ── 走原有全流程（无自定义 mask）──────────────────────
        if mask_override is None:
            print("🔵 全流程修复（run_single_test）")
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
                mask_erode_ksize=9,
                mask_erode_iters=1,
                mask_smooth_sigma=15,
            )
            print("✅ 修复完成")
            return result

        # ── 使用自定义 mask（跳过检测 + SAM）─────────────────
        print(f"🔵 自定义 Mask 修复（{mask_override}）")
        try:
            # 1. 加载原图与 mask
            image_pil = Image.open(image_path).convert("RGB")
            W, H = image_pil.size

            mask_pil = Image.open(mask_override).convert("L")
            if mask_pil.size != (W, H):
                mask_pil = mask_pil.resize((W, H), Image.NEAREST)

            # 2. 轻度平滑 mask 边缘，减少修复接缝
            mask_smooth = mask_pil.filter(ImageFilter.GaussianBlur(radius=4))

            # 3. object_removal 时强制空白提示词，避免扩散模型生成多余内容
            prompt = "" if task_type == "object_removal" else inpaint_prompt
            negative_prompt = (
                "artifacts, blurry, ugly, distorted, text, watermark, "
                "speech bubble, logo"
            )

            # 4. SDXL Inpainting
            sdxl_w, sdxl_h = self._sdxl_size(W, H)
            result_img = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image_pil.resize((sdxl_w, sdxl_h), Image.LANCZOS),
                mask_image=mask_smooth.resize((sdxl_w, sdxl_h), Image.NEAREST),
                guidance_scale=7.5,
                num_inference_steps=30,
                strength=0.99,
            ).images[0]

            # 5. 还原到原始尺寸
            result_img = result_img.resize((W, H), Image.LANCZOS)

            # 6. 保存输出
            output_dir = os.path.join("gradio_outputs", "custom_mask")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir, "grounded_sam_inpainting_output.jpg"
            )
            result_img.save(output_path, quality=95)

            print("✅ 自定义 Mask 修复完成")
            return {"status": "success", "output_dir": output_dir}

        except Exception as e:
            print(f"❌ 自定义 Mask 修复失败: {e}")
            return {"status": "failed", "error": str(e)}

    @staticmethod
    def _sdxl_size(W: int, H: int) -> tuple[int, int]:
        """
        将原图尺寸向下对齐到 SDXL 要求的 8 的倍数，
        同时不超过 1024，保持宽高比。
        """
        MAX = 1024
        scale = min(MAX / W, MAX / H, 1.0)
        new_w = (int(W * scale) // 8) * 8
        new_h = (int(H * scale) // 8) * 8
        return max(new_w, 8), max(new_h, 8)