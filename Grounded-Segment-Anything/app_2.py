"""
漫画图像编辑系统 — 高端版
支持连通域选择 + 精准重绘
"""

import gradio as gr
from model_api import InpaintModel
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from scipy import ndimage

# ─────────────────────────────────────────────
#  全局配置
# ─────────────────────────────────────────────

# 每个连通域的高亮配色（RGBA）
REGION_COLORS = [
    (99,  179, 237, 180),   # Sky blue
    (252, 129,  74, 180),   # Coral
    (104, 211, 145, 180),   # Mint
    (214, 158, 246, 180),   # Lavender
    (246, 224,  94, 180),   # Gold
    (99,  230, 190, 180),   # Teal
    (250, 112, 154, 180),   # Pink
    (144, 205, 244, 180),   # Ice
]

model = InpaintModel()

# ─────────────────────────────────────────────
#  运行时状态缓存（避免重复计算）
# ─────────────────────────────────────────────
_state = {
    "labeled": None,
    "num_features": 0,
    "original": None,
    "raw_mask": None,
}


# ─────────────────────────────────────────────
#  工具函数
# ─────────────────────────────────────────────

def _to_binary(mask_img: Image.Image) -> np.ndarray:
    arr = np.array(mask_img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return (arr > 128).astype(np.uint8)


def _label_mask(mask_img: Image.Image):
    binary = _to_binary(mask_img)
    labeled, n = ndimage.label(binary)
    return labeled, n


def _colorize_overview(orig: Image.Image, labeled: np.ndarray, n: int,
                        highlight: list[int] | None = None) -> Image.Image:
    """在原图上叠加所有（或指定）连通域的彩色高亮。"""
    base = orig.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))

    targets = highlight if highlight is not None else list(range(1, n + 1))
    for i in targets:
        color = REGION_COLORS[(i - 1) % len(REGION_COLORS)]
        region_mask_arr = ((labeled == i).astype(np.uint8) * 255)
        region_mask_img = Image.fromarray(region_mask_arr, mode="L")
        colored_layer = Image.new("RGBA", base.size, color)
        overlay.paste(colored_layer, mask=region_mask_img)

    return Image.alpha_composite(base, overlay).convert("RGB")


def _make_thumbnail(orig: Image.Image, labeled: np.ndarray, idx: int,
                    size: int = 220) -> Image.Image:
    """裁剪并高亮单个连通域缩略图，带编号标签。"""
    color = REGION_COLORS[(idx - 1) % len(REGION_COLORS)]
    comp_mask = labeled == idx

    rows = np.where(np.any(comp_mask, axis=1))[0]
    cols = np.where(np.any(comp_mask, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return Image.new("RGB", (size, size), (20, 20, 35))

    pad = 18
    h, w = np.array(orig).shape[:2]
    r0, r1 = max(0, rows[0] - pad), min(h, rows[-1] + pad)
    c0, c1 = max(0, cols[0] - pad), min(w, cols[-1] + pad)

    # 灰化背景，彩色高亮目标区域
    orig_arr = np.array(orig.convert("RGB"))
    gray = np.mean(orig_arr, axis=2, keepdims=True).repeat(3, axis=2).astype(np.uint8)
    result = gray.copy()
    alpha = color[3] / 255.0
    crop_mask = comp_mask  # full image mask
    for c in range(3):
        result[:, :, c] = np.where(
            crop_mask,
            np.clip((1 - alpha) * orig_arr[:, :, c] + alpha * color[c], 0, 255).astype(np.uint8),
            gray[:, :, c]
        )

    cropped = Image.fromarray(result[r0:r1, c0:c1])

    # 缩放到正方形缩略图
    thumb = cropped.copy()
    thumb.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (14, 14, 24))
    offset = ((size - thumb.width) // 2, (size - thumb.height) // 2)
    canvas.paste(thumb, offset)

    # 绘制编号角标
    draw = ImageDraw.Draw(canvas)
    badge_r = 14
    draw.ellipse([4, 4, 4 + badge_r * 2, 4 + badge_r * 2],
                 fill=tuple(color[:3]))
    draw.text((4 + badge_r, 4 + badge_r), str(idx),
              fill=(10, 10, 20), anchor="mm")

    return canvas


def merge_masks(labeled: np.ndarray, indices: list[int]) -> Image.Image:
    """将选中区域合并为一张二值 Mask。"""
    merged = np.zeros_like(labeled, dtype=np.uint8)
    for i in indices:
        merged[labeled == i] = 255
    return Image.fromarray(merged, mode="L")


# ─────────────────────────────────────────────
#  Step 1 — 检测 + 分割 + 连通域分析
# ─────────────────────────────────────────────

def detect_fn(image, det_prompt):
    if image is None:
        return (None, None, [], gr.update(choices=[], value=[]),
                status_html("error", "请先上传图片"))

    try:
        input_path = "temp_input.jpg"
        image.save(input_path)

        raw_mask = model.get_mask(input_path, det_prompt)
        labeled, n = _label_mask(raw_mask)

        _state["labeled"] = labeled
        _state["num_features"] = n
        _state["original"] = image
        _state["raw_mask"] = raw_mask

        overview = _colorize_overview(image, labeled, n)
        thumbs = [(_make_thumbnail(image, labeled, i), f"区域 {i}") for i in range(1, n + 1)]
        choices = [f"区域 {i}" for i in range(1, n + 1)]

        return (
            raw_mask,
            overview,
            thumbs,
            gr.update(choices=choices, value=choices),
            status_html("success", f"检测完成 — 共发现 <b>{n}</b> 个连通区域"),
        )
    except Exception as e:
        return (None, None, [], gr.update(choices=[], value=[]),
                status_html("error", f"检测失败：{e}"))


# ─────────────────────────────────────────────
#  Step 1b — 勾选变化时实时预览选中区域
# ─────────────────────────────────────────────

def preview_selection(selected_labels):
    if _state["labeled"] is None or _state["original"] is None:
        return None

    indices = [int(s.split(" ")[1]) for s in selected_labels if s.startswith("区域 ")]
    if not indices:
        return _state["original"]

    return _colorize_overview(_state["original"], _state["labeled"],
                               _state["num_features"], highlight=indices)


# ─────────────────────────────────────────────
#  Step 2 — 修复
# ─────────────────────────────────────────────

def inpaint_fn(image, det_prompt, inpaint_prompt, task_type, selected_labels):
    if image is None:
        return None, status_html("error", "请先上传图片")
    if _state["labeled"] is None:
        return None, status_html("error", "请先执行检测，生成 Mask")
    if not selected_labels:
        return None, status_html("error", "请至少勾选一个区域")

    try:
        input_path = "temp_input.jpg"
        image.save(input_path)

        indices = [int(s.split(" ")[1]) for s in selected_labels]
        selected_mask = merge_masks(_state["labeled"], indices)
        mask_path = "temp_selected_mask.png"
        selected_mask.save(mask_path)

        # ⚠️  model_api.infer 需支持 mask_override 参数
        #     在 model_api.py 中增加：
        #       def infer(self, image_path, det_prompt, inpaint_prompt,
        #                 task_type, mask_override=None):
        #           if mask_override: mask = Image.open(mask_override)
        #           else: mask = self.get_mask(...)
        result = model.infer(
            input_path,
            det_prompt,
            inpaint_prompt,
            task_type,
            mask_override=mask_path,   # 新增参数
        )

        if result["status"] != "success":
            return None, status_html("error", "修复管线返回失败状态")

        output_path = os.path.join(
            result["output_dir"],
            "grounded_sam_inpainting_output.jpg"
        )
        return Image.open(output_path), status_html("success", "修复完成！")

    except Exception as e:
        return None, status_html("error", f"修复失败：{e}")


# ─────────────────────────────────────────────
#  状态标签 HTML 辅助
# ─────────────────────────────────────────────

def status_html(kind: str, msg: str) -> str:
    palette = {
        "success": ("#0f2a1a", "#34d399", "✓"),
        "error":   ("#2a0f0f", "#f87171", "✗"),
        "info":    ("#0f1a2a", "#60a5fa", "i"),
    }
    bg, fg, icon = palette.get(kind, palette["info"])
    return (
        f'<div style="background:{bg};border:1px solid {fg}33;border-radius:8px;'
        f'padding:10px 14px;color:{fg};font-size:13px;font-weight:500;'
        f'letter-spacing:.3px;display:flex;align-items:center;gap:8px;">'
        f'<span style="font-weight:700">{icon}</span>{msg}</div>'
    )


# ─────────────────────────────────────────────
#  CSS — 暗色高端主题
# ─────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── 全局重置 ── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #080810 !important;
    font-family: 'Outfit', sans-serif !important;
    color: #c8c8e0 !important;
}

/* ── 页面宽度 ── */
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; padding: 0 24px !important; }

/* ── 顶部标题区 ── */
#header {
    padding: 48px 0 32px;
    border-bottom: 1px solid #1e1e32;
    margin-bottom: 32px;
}
#header h1 {
    font-size: 28px;
    font-weight: 700;
    color: #eeeeff;
    letter-spacing: -0.5px;
    margin: 0 0 6px;
}
#header p {
    font-size: 14px;
    color: #6666aa;
    margin: 0;
    letter-spacing: .3px;
}

/* ── 步骤徽章 ── */
.step-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #7b7bcc;
    margin-bottom: 14px;
}
.step-badge::before {
    content: '';
    display: block;
    width: 18px;
    height: 1px;
    background: #4444aa;
}

/* ── 卡片面板 ── */
.panel {
    background: #10101e;
    border: 1px solid #1e1e38;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 20px;
}

/* ── 图像组件 ── */
.gradio-image, .svelte-1ipelgc {
    border-radius: 10px !important;
    border: 1px solid #1e1e38 !important;
    background: #0c0c1a !important;
    overflow: hidden !important;
}
.gradio-image img { border-radius: 9px !important; }

/* ── 输入框 ── */
textarea, input[type=text] {
    background: #0c0c1a !important;
    border: 1px solid #252545 !important;
    border-radius: 8px !important;
    color: #d0d0f0 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
    transition: border-color .2s !important;
}
textarea:focus, input[type=text]:focus {
    border-color: #5555cc !important;
    outline: none !important;
    box-shadow: 0 0 0 3px #3333aa22 !important;
}

/* ── 标签文字 ── */
label > span, .label-wrap span {
    font-size: 12px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: .8px !important;
    color: #6666aa !important;
}

/* ── 主按钮 ── */
#detect-btn, #inpaint-btn {
    background: linear-gradient(135deg, #3a3acc 0%, #5858ee 100%) !important;
    border: none !important;
    border-radius: 9px !important;
    color: #fff !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: .5px !important;
    padding: 11px 22px !important;
    cursor: pointer !important;
    transition: opacity .2s, transform .1s !important;
    box-shadow: 0 4px 20px #3333cc33 !important;
}
#detect-btn:hover, #inpaint-btn:hover { opacity: .88 !important; transform: translateY(-1px) !important; }
#detect-btn:active, #inpaint-btn:active { transform: translateY(0) !important; }

#inpaint-btn {
    background: linear-gradient(135deg, #1a6644 0%, #22aa66 100%) !important;
    box-shadow: 0 4px 20px #22aa6633 !important;
}

/* ── Radio 任务选择 ── */
.gradio-radio label {
    background: #10101e !important;
    border: 1px solid #252545 !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    margin-right: 8px !important;
    cursor: pointer !important;
    transition: border-color .2s !important;
}
.gradio-radio input[type=radio]:checked + label,
.gradio-radio label:has(input:checked) {
    border-color: #5555cc !important;
    background: #18183a !important;
    color: #aaaaff !important;
}

/* ── CheckboxGroup ── */
.gradio-checkboxgroup label {
    background: #0e0e1e !important;
    border: 1px solid #1e1e38 !important;
    border-radius: 8px !important;
    padding: 6px 14px !important;
    margin: 3px !important;
    font-size: 13px !important;
    cursor: pointer !important;
    transition: border-color .2s, background .2s !important;
}
.gradio-checkboxgroup label:hover { border-color: #5555cc !important; }
.gradio-checkboxgroup input[type=checkbox]:checked + label,
.gradio-checkboxgroup label:has(input:checked) {
    border-color: #5555cc !important;
    background: #18183a !important;
    color: #aaaaff !important;
}

/* ── Gallery ── */
.gradio-gallery { background: transparent !important; }
.gradio-gallery .thumbnail-item {
    border-radius: 10px !important;
    border: 1px solid #1e1e38 !important;
    overflow: hidden !important;
    transition: border-color .2s, transform .15s !important;
}
.gradio-gallery .thumbnail-item:hover {
    border-color: #5555cc !important;
    transform: translateY(-2px) !important;
}
.gradio-gallery .thumbnail-item.selected {
    border-color: #7777ff !important;
    box-shadow: 0 0 0 2px #3333aa55 !important;
}

/* ── 分割线 ── */
hr { border: none; border-top: 1px solid #1a1a30; margin: 24px 0; }

/* ── 滚动条 ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a12; }
::-webkit-scrollbar-thumb { background: #2a2a50; border-radius: 3px; }

/* ── 进度条 ── */
.progress-bar { background: #3a3acc !important; }
.eta-bar { background: #1e1e38 !important; }
"""


# ─────────────────────────────────────────────
#  UI 布局
# ─────────────────────────────────────────────

with gr.Blocks(css=CSS, title="漫画编辑系统") as demo:

    # ── 标题 ──
    gr.HTML("""
    <div id="header">
        <h1>✦ 漫画图像编辑系统</h1>
        <p>Grounded-SAM · 连通域精准选择 · 扩散模型修复</p>
    </div>
    """)

    # ══════════════════════════════════════════
    #  STEP 1 — 上传 & 检测
    # ══════════════════════════════════════════
    gr.HTML('<div class="step-badge">Step 01 — 上传图片 & 生成 Mask</div>')

    with gr.Row(equal_height=False):
        # 左列：上传 + 参数
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="原始图片", height=320)
            det_prompt = gr.Textbox(
                value="speech bubble",
                label="检测提示词",
                placeholder="描述要检测的目标，如 speech bubble / person / car …"
            )
            detect_btn = gr.Button("⊕  生成 Mask + 分析区域", elem_id="detect-btn")
            status_box = gr.HTML(label="状态")

        # 右列：原始 Mask + 总览
        with gr.Column(scale=1):
            raw_mask_img = gr.Image(label="原始 Mask（二值）", height=220)
            overview_img = gr.Image(label="连通域总览（彩色标注）", height=320)

    # ══════════════════════════════════════════
    #  STEP 2 — 选择连通域
    # ══════════════════════════════════════════
    gr.HTML('<hr><div class="step-badge">Step 02 — 选择目标区域</div>')

    region_gallery = gr.Gallery(
        label="各连通域预览（点击可查看大图）",
        columns=6,
        height=260,
        object_fit="cover",
        allow_preview=True,
        format="jpeg",
    )

    region_checkboxes = gr.CheckboxGroup(
        choices=[],
        value=[],
        label="勾选要修复的区域（可多选）",
        interactive=True,
    )

    selection_preview = gr.Image(label="当前选中区域预览", height=300)

    # ══════════════════════════════════════════
    #  STEP 3 — 修复参数 & 执行
    # ══════════════════════════════════════════
    gr.HTML('<hr><div class="step-badge">Step 03 — 执行修复</div>')

    with gr.Row():
        with gr.Column(scale=2):
            inpaint_prompt = gr.Textbox(
                value="clean background, manga style",
                label="修复提示词",
                placeholder="描述修复目标，如 clean background / sunny landscape …"
            )
        with gr.Column(scale=1):
            task_type = gr.Radio(
                choices=["object_removal", "object_replacement"],
                value="object_removal",
                label="任务类型",
            )

    with gr.Row():
        inpaint_btn = gr.Button("◈  执行修复", elem_id="inpaint-btn", scale=1)
        inpaint_status = gr.HTML(scale=2)

    output_image = gr.Image(label="修复结果", height=420)

    # ══════════════════════════════════════════
    #  事件绑定
    # ══════════════════════════════════════════

    # 检测按钮
    detect_btn.click(
        fn=detect_fn,
        inputs=[input_image, det_prompt],
        outputs=[raw_mask_img, overview_img, region_gallery,
                 region_checkboxes, status_box],
        show_progress=True,
    )

    # 勾选变化 → 实时更新选中预览
    region_checkboxes.change(
        fn=preview_selection,
        inputs=[region_checkboxes],
        outputs=[selection_preview],
    )

    # 修复按钮
    inpaint_btn.click(
        fn=inpaint_fn,
        inputs=[input_image, det_prompt, inpaint_prompt,
                task_type, region_checkboxes],
        outputs=[output_image, inpaint_status],
        show_progress=True,
    )


# ─────────────────────────────────────────────
#  启动
# ─────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        debug=True,
    )