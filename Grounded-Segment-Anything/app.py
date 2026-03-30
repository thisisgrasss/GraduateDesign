import gradio as gr
import torch
from PIL import Image
import numpy as np

# 从你的核心脚本导入
from grounded_sam_inpainting_v4_final import run_single_test, load_models

# 1. 启动时加载模型（全局单例）
print("系统初始化中...")
MODELS_DICT = load_models(device="cuda" if torch.cuda.is_available() else "cpu")

def process_comic(input_img, task, prompt, box_thresh, text_thresh, dilation):
    if input_img is None:
        return None, "请先上传图片"
    
    # 调用你核心脚本的推理函数
    # 这里的参数名需要对应你 run_single_test 里的定义
    try:
        result_img, metrics = run_single_test(
            image_pil=input_img,
            task_type=task,
            text_prompt=prompt,
            box_threshold=box_thresh,
            text_threshold=text_thresh,
            dsp_dilate_ksize=int(dilation),
            **MODELS_DICT
        )
        return result_img, metrics
    except Exception as e:
        return None, f"错误: {str(e)}"

# 2. 构建 Gradio 界面
with gr.Blocks(title="漫画 AI 修复工具") as demo:
    gr.Markdown("## 🎨 漫画图像修复控制台 (Grounded-SAM + SDXL/LaMa)")
    gr.Markdown("针对漫画场景优化的 **移除(Removal)** 与 **替换(Replacement)** 流程。")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="上传漫画")
            
            task_type = gr.Radio(
                choices=["object_removal", "object_replacement"], 
                value="object_removal", 
                label="任务模式"
            )
            
            # 只有在替换模式下才显示 Prompt
            prompt_input = gr.Textbox(
                label="Prompt", 
                placeholder="想要替换成什么？(例如: a cute cat)",
                visible=False # 初始设为隐藏
            )
            
            # 动态显示/隐藏 Prompt 框
            task_type.change(
                fn=lambda x: gr.update(visible=(x == "object_replacement")),
                inputs=[task_type],
                outputs=[prompt_input]
            )
            
            with gr.Accordion("精细调优参数", open=False):
                box_threshold = gr.Slider(0.1, 0.8, value=0.3, step=0.05, label="SAM 检出阈值")
                dilation_size = gr.Slider(1, 51, value=15, step=2, label="Mask 膨胀 (Dilation)")

            run_button = gr.Button("🚀 开始执行", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="修复结果")
            metrics_display = gr.JSON(label="质量指标 (Metrics)")

    # 绑定事件
    run_button.click(
        fn=process_comic,
        inputs=[input_image, task_type, prompt_input, box_threshold, 0.25, dilation_size],
        outputs=[output_image, metrics_display]
    )

# 3. 运行
if __name__ == "__main__":
    # share=True 可以生成一个临时的公网访问链接
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)