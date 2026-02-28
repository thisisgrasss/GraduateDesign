import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt


# diffusers
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionXLInpaintPipeline

# 改动6：新增 NMS 导入，用于过滤重复检测框
from torchvision.ops import nms


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []

    scores = []  # 改动6：收集每个框的置信度分数，供 NMS 使用
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        score = logit.max().item()
        scores.append(score)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(score)[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    # for logit, box in zip(logits_filt, boxes_filt):
    #     pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
    #     if with_logits:
    #         pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
    #     else:
    #         pred_phrases.append(pred_phrase)

    # return boxes_filt, pred_phrases
    return boxes_filt, pred_phrases, torch.tensor(scores)


# 改动6：新增 NMS 过滤函数，去除重叠检测框
def filter_boxes_nms(boxes, phrases, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return boxes, phrases, scores
    # nms 要求 xyxy 格式，boxes_filt 此时已是 cx,cy,w,h，需先转换
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
    keep = nms(boxes_xyxy, scores, iou_threshold)
    return boxes[keep], [phrases[i] for i in keep], scores[keep]

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)

# 改动2：新增保持宽高比的 resize 函数，避免图像形变
def resize_with_padding(img, target_size=512):
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    pad_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    pad_img.paste(img_resized, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    return pad_img, (new_w, new_h), ((target_size - new_w) // 2, (target_size - new_h) // 2)

def restore_from_padding(img, original_size, new_wh, offset, target_size=512):
    """裁剪掉 padding，还原到原始尺寸"""
    ox, oy = offset
    nw, nh = new_wh
    img_cropped = img.crop((ox, oy, ox + nw, oy + nh))
    return img_cropped.resize(original_size, Image.LANCZOS)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--det_prompt", type=str, required=True, help="text prompt")
    parser.add_argument("--inpaint_prompt", type=str, required=True, help="inpaint prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="save your huggingface large model cache")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--inpaint_mode", type=str, default="first", help="inpaint mode")
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    image_path = args.input_image
    det_prompt = args.det_prompt
    inpaint_prompt = args.inpaint_prompt
    output_dir = args.output_dir
    cache_dir=args.cache_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    inpaint_mode = args.inpaint_mode
    device = args.device

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # run grounding dino model
    # 改动6：get_grounding_output 现在额外返回 scores，用于 NMS
    boxes_filt, pred_phrases, scores = get_grounding_output(
        model, image, det_prompt, box_threshold, text_threshold, device=device
    )

    # 改动6：在坐标转换前，先用 NMS 过滤重叠框（此时仍是归一化 cx,cy,w,h 格式）
    boxes_filt, pred_phrases, scores = filter_boxes_nms(boxes_filt, pred_phrases, scores, iou_threshold=0.5)

    # initialize SAM
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )

    # masks: [1, 1, 512, 512]

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "grounded_sam_output.jpg"), bbox_inches="tight")

    # ✅ 改动4：多目标 mask 合并逻辑更健壮，使用 torch.any 替代 torch.sum
    if inpaint_mode == 'merge':
        merged_mask = torch.any(masks, dim=0)[0].cpu().numpy().astype(np.uint8) * 255
    else:
        merged_mask = masks[0][0].cpu().numpy().astype(np.uint8) * 255


    
    # ✅ 改动1：对 mask 做膨胀 + 高斯模糊，软化边缘，消除 inpainting 接缝感
    # 原代码：mask_pil = Image.fromarray(mask)，直接使用原始二值 mask，边缘生硬
    kernel = np.ones((15, 15), np.uint8)
    merged_mask = cv2.dilate(merged_mask, kernel, iterations=1)   # 膨胀，防止边缘残留
    merged_mask = cv2.GaussianBlur(merged_mask, (21, 21), 0)      # 高斯模糊，软化边缘

    mask_pil = Image.fromarray(merged_mask)
    image_pil = Image.fromarray(image)



    # ✅ 改动5：更换为效果更强的 SD 2 inpainting 模型
    # 原代码：from_pretrained("runwayml/stable-diffusion-inpainting", ...)
    # pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-2-inpainting",   # ← 改动5：换用更强模型
    #     torch_dtype=torch.float16,
    #     cache_dir=cache_dir
    # )
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    cache_dir=cache_dir
    )

    pipe = pipe.to("cuda")

    # ✅ 改动2：用保持宽高比的 resize 替代强制拉伸到 512x512
    # 原代码：image_pil = image_pil.resize((512, 512))
    #         mask_pil  = mask_pil.resize((512, 512))
    original_size = image_pil.size  # 保存原始尺寸用于最终还原
    image_pil_padded, new_wh, offset = resize_with_padding(image_pil, target_size=1024)
    mask_pil_padded, _, _ = resize_with_padding(mask_pil.convert("RGB"), target_size=1024)
    mask_pil_padded = mask_pil_padded.convert("L")  # 保持单通道

    # ✅ 改动3：增加 negative_prompt、num_inference_steps、guidance_scale、strength 参数
    # 原代码：image = pipe(prompt=inpaint_prompt, image=image_pil, mask_image=mask_pil).images[0]
    result = pipe(
        prompt=inpaint_prompt,
        negative_prompt="blurry, bad quality, distorted, artifacts, ugly, low resolution",  # ← 改动3
        image=image_pil_padded,
        mask_image=mask_pil_padded,
        num_inference_steps=50,    # ← 改动3：默认通常是20步，提升到50步质量更好
        guidance_scale=7.5,        # ← 改动3：控制 prompt 引导强度
        strength=0.99,             # ← 改动3：接近1为完全重绘 mask 区域
    ).images[0]

    # ✅ 改动2：还原 padding，resize 回原始尺寸
    # 原代码：image = image.resize(size)
    result = restore_from_padding(result, original_size, new_wh, offset, target_size=1024)

    result.save(os.path.join(output_dir, "grounded_sam_inpainting_output.jpg"))
    print(f"Inpainting result saved to {os.path.join(output_dir, 'grounded_sam_inpainting_output.jpg')}")
