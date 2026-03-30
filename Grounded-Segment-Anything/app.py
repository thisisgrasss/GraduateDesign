import gradio as gr
from model_api import InpaintModel
from PIL import Image
import os

print("🚀 启动 Gradio...")

# ⚠️ 全局加载模型（只执行一次）
model = InpaintModel()


# ===== Step1: 检测 =====
def detect_fn(image, det_prompt):
    print("👉 点击：生成 Mask")

    if image is None:
        print("❌ 没有输入图片")
        return None, None

    try:
        input_path = "temp_input.jpg"
        image.save(input_path)

        mask = model.get_mask(input_path, det_prompt)

        return image, mask

    except Exception as e:
        print("❌ detect 错误:", e)
        return None, None


# ===== Step2: 修复 =====
def inpaint_fn(image, det_prompt, inpaint_prompt, task_type):
    print("👉 点击：执行修复")

    if image is None:
        print("❌ 没有输入图片")
        return None

    try:
        input_path = "temp_input.jpg"
        image.save(input_path)

        result = model.infer(
            input_path,
            det_prompt,
            inpaint_prompt,
            task_type
        )

        if result["status"] != "success":
            print("❌ 修复失败")
            return None

        output_path = os.path.join(
            result["output_dir"],
            "grounded_sam_inpainting_output.jpg"
        )

        return Image.open(output_path)

    except Exception as e:
        print("❌ inpaint 错误:", e)
        return None


# ===== UI =====
with gr.Blocks() as demo:

    gr.Markdown("## 🎨 漫画图像编辑系统（可解释交互版）")

    with gr.Row():
        input_image = gr.Image(type="pil", label="原图")
        mask_image = gr.Image(label="Mask")
        output_image = gr.Image(label="修复结果")

    det_prompt = gr.Textbox(value="speech bubble", label="检测提示词")
    inpaint_prompt = gr.Textbox(value="clean background", label="修复提示词")

    task_type = gr.Radio(
        ["object_removal", "object_replacement"],
        value="object_removal",
        label="任务类型"
    )

    with gr.Row():
        detect_btn = gr.Button("① 生成 Mask")
        inpaint_btn = gr.Button("② 执行修复")

    detect_btn.click(
        fn=detect_fn,
        inputs=[input_image, det_prompt],
        outputs=[input_image, mask_image],
        show_progress=True
    )

    inpaint_btn.click(
        fn=inpaint_fn,
        inputs=[input_image, det_prompt, inpaint_prompt, task_type],
        outputs=output_image,
        show_progress=True
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        debug=True   # ⭐ 必开！否则你看不到报错
    )