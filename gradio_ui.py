import os
import shutil
import torch
from torch.nn import functional as F
import torchvision.transforms as T
import gradio as gr
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

weights = os.listdir("weights")
weights = [os.path.join("weights", w) for w in weights]
scales = [0.25, 0.5, 1.0, 2.0, 4.0]
model = None

def load_checkpoint(union, checkpoint):
    shutil.rmtree("./train_log", ignore_errors=True)
    shutil.copytree(checkpoint, "./train_log")
    checkpoint = "train_log"
    try:
        del Model
    except:
        pass
    global model
    if union:
        from model.GMFSS_infer_u import Model
    else:
        from model.GMFSS_infer_b import Model
    model = Model()
    try:
        model.load_model(checkpoint, -1)
    except:
        gr.Info("Model Loading Failed. Maybe the weights are not for this model?")
    if union:
        gr.Info("Loaded Union Model")
    else:
        gr.Info("Loaded Non-union Model")
    model.eval()
    model.device()
    return

def infer(frame1, frame2, scale, exp):
    img0 = np.array(frame1)
    img1 = np.array(frame2)
    img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    n, c, h, w = img0.shape
    tmp = max(64, int(64 / scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    img0 = F.interpolate(img0, (ph, pw), mode='bilinear', align_corners=False)
    img1 = F.interpolate(img1, (ph, pw), mode='bilinear', align_corners=False)
    multi = 2 ** exp
    reuse_things = model.reuse(img0, img1, scale)
    res = []
    for i in range(multi - 1):
        res.append(model.inference(img0, img1, reuse_things, (i+1) * 1.0 / multi))
    ret = []
    transform = T.ToPILImage()
    for image in res:
        image = F.interpolate(image, (h, w), mode='bilinear', align_corners=False)
        image = (((image[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0))[:h, :w])
        image = transform(image)
        ret.append(image)
    return res

with gr.Blocks() as demo:
    gr.Markdown(
    '''
    # GMFSS Inference
    ''')
    with gr.Column():
        union = gr.Checkbox(label="Union", info="Use Union Model", value=True)
        checkpoint = gr.Dropdown(choices=weights, label="Checkpoint")
        checkpoint_btn = gr.Button("Load")

        scale = gr.Dropdown(choices=scales, value=1.0, label="Scale", info="Try scale=0.5 for 4k video")
        exp = gr.Number(minimum=0, value=1, label="Exp")
        with gr.Row():
            frame1 = gr.Image(type="pil", label="Frame 1", sources=['upload'])
            frame2 = gr.Image(type="pil", label="Frame 2", sources=['upload'])
    with gr.Column():
        frameI = gr.Gallery(type="pil", label="Synthesized Result")
        run_btn = gr.Button(value="Run")

    checkpoint_btn.click(fn=load_checkpoint, inputs=[union, checkpoint], outputs=None)
    run_btn.click(fn=infer, inputs=[frame1, frame2, scale, exp], outputs=[frameI])
demo.launch(server_name="0.0.0.0", server_port=7068)
