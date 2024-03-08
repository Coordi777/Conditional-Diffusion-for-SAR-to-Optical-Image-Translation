import gradio as gr
import argparse
import os

import pandas as pd
from PIL import Image
import numpy as np
import torch as th
from torchvision import transforms

from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


transform_opt = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

])

device = th.device('cuda:3' if th.cuda.is_available() else 'cpu')
args = create_argparser().parse_args()

print("creating model and diffusion...")
model, _ = create_model_and_diffusion(
    **args_to_dict(args, model_and_diffusion_defaults().keys())
)

model.load_state_dict(
    dist_util.load_state_dict('/attached/remote-home2/bxy/myDiff/Guided_diffusion/results_sen12_nosie/ema_0.9999_640000.pt',
                              map_location="cpu")
)
model.to(device)
model.eval()

model_kwargs = {}


def predict(condition, timestep_respacing, model_name):
    # 更新args的timestep_respacing参数
    if model_name == "Sent":
        model.load_state_dict(
            dist_util.load_state_dict(
                '/attached/remote-home2/bxy/myDiff/Guided_diffusion/results_sen12_nosie/ema_0.9999_640000.pt',
                map_location="cpu")
        )
        model.to(device)
        model.eval()
        args.diffusion_steps = 4000
        args.predict_xstart = False
        args.timestep_respacing = str(timestep_respacing)
        _, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
    elif model_name == "GF3":
        model.load_state_dict(
            dist_util.load_state_dict(
                '/attached/remote-home2/bxy/myDiff/Guided_diffusion/results_xstart/ema_0.9999_140000.pt',
                map_location="cpu")
        )
        model.to(device)
        model.eval()
        args.predict_xstart = True
        args.timestep_respacing = str(timestep_respacing)
        args.diffusion_steps = 4000
        _, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )

    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )
    condition = transform_opt(condition)
    condition = condition.unsqueeze(0)
    condition = condition.to(device)
    sample = sample_fn(
        model,
        (1, 3, 256, 256),
        clip_denoised=True,
        model_kwargs=model_kwargs,
        noise=None,
        condition=condition,
        progress=gr.Progress()
    )

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.cpu().numpy()
    sample = sample.squeeze(0)
    sample = Image.fromarray(sample)
    return sample


demo = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil"), gr.Slider(25, 4000),
            gr.Radio(["Sent", "GF3"], label="Model", info="Which model to you want to use?"), ],
    outputs=gr.Image(type="pil", shape=(256, 256)),
    examples=[
        [os.path.join(os.path.dirname(__file__), "sar_1.png"), 250, "Sent"],
        [os.path.join(os.path.dirname(__file__), "sar_2.png"), 500, "Sent"],
        [os.path.join(os.path.dirname(__file__), "sar_3.png"), 1000, "Sent"],
        [os.path.join(os.path.dirname(__file__), "sar_4.png"), 2000, "Sent"],
    ],
    title="SAR to Optical Image🚀",
    description="""
        # 🎯 Instruction
        This is a project that converts SAR images into optical images, based on conditional diffusion. 

        Input a SAR image, and its corresponding optical image will be obtained.

        ## 📢 Inputs
        - `condition`: the SAR image that you want to transfer.
        - `timestep_respacing`: the number of iteration steps when inference.

        ## 🎉 Outputs
        - The corresponding optical image.
        
        **Paper** : [Guided Diffusion for Image Generation](https://arxiv.org/abs/2105.05233)
        
        **Github** : https://github.com/Coordi777/Conditional_SAR2OPT
    """
)

if __name__ == "__main__":
    demo.launch(server_port=16006)
