"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch as th
import torch.distributed as dist
from torchvision import transforms

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from tqdm.auto import tqdm

transform_opt = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

reverse_transforms = transforms.Compose([
    transforms.Lambda(lambda t: (t + 1) / 2),
    transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
    transforms.Lambda(lambda t: t * 255.),
    transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
    transforms.ToPILImage(),
])

class pair_Dataset(Dataset):
    def __init__(self, path, transforms_sar, transforms_opt):
        self.path_sar = os.path.join(path, "sar")
        self.path_opt = os.path.join(path, "opt")
        self.trans_sar = transforms_sar
        self.trans_opt = transforms_opt
        sar_name_list = os.listdir(self.path_sar)
        opt_name_list = os.listdir(self.path_opt)

        self.all_img_sar = [os.path.join(self.path_sar, sar_name_list[i]) for i in range(len(sar_name_list))]
        self.all_img_opt = [os.path.join(self.path_opt, opt_name_list[i]) for i in range(len(opt_name_list))]

    def __getitem__(self, index):
        img_path_sar = self.all_img_sar[index]
        img_path_opt = self.all_img_opt[index]
        pil_img_sar = Image.open(img_path_sar).convert('RGB')
        pil_img_opt = Image.open(img_path_opt).convert('RGB')
        img_sar = self.trans_sar(pil_img_sar)
        img_opt = self.trans_opt(pil_img_opt)
        return img_sar, img_opt

    def __len__(self):
        return len(self.all_img_sar)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log(f"sampling with steps {args.timestep_respacing}...")
    all_images = []
    all_labels = []
    all_sar = []
    all_opt = []
    # TODO: change the path
    pair_datas = pair_Dataset('', transform_opt, transform_opt)
    data_loader = DataLoader(pair_datas, batch_size=args.batch_size, shuffle=False, num_workers=0)
    count = 0
    if not os.path.exists(f'sample_results/{args.timestep_respacing}'):
        os.makedirs(f'sample_results/{args.timestep_respacing}')
        os.makedirs(f'sample_results/{args.timestep_respacing}/gen_opt')
        os.makedirs(f'sample_results/{args.timestep_respacing}/cond_sar')
        os.makedirs(f'sample_results/{args.timestep_respacing}/gt_opt')
    loader = iter(data_loader)
    while count * args.batch_size < args.num_samples:
        model_kwargs = {}
        all_images = []
        all_labels = []
        all_sar = []
        all_opt = []
        condition, gt = next(loader)
        condition, gt = condition.to(dist_util.dev()), gt.to(dist_util.dev())
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            noise=None,
            condition=condition,
            progress=True
        )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        condition = ((condition + 1) * 127.5).clamp(0, 255).to(th.uint8)
        condition = condition.permute(0, 2, 3, 1).squeeze()
        condition = condition.contiguous()

        img_opt = ((gt + 1) * 127.5).clamp(0, 255).to(th.uint8)
        img_opt = img_opt.permute(0, 2, 3, 1)
        img_opt = img_opt.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        gathered_sar = [th.zeros_like(condition) for _ in range(dist.get_world_size())]
        gt_opt = [th.zeros_like(img_opt) for _ in range(dist.get_world_size())]

        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        dist.all_gather(gathered_sar, condition)
        dist.all_gather(gt_opt, img_opt)

        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        all_sar.extend([condition.cpu().numpy() for condition in gathered_sar])
        all_opt.extend([img_opt.cpu().numpy() for img_opt in gt_opt])

        arr = np.concatenate(all_images, axis=0)
        arr_sar = np.concatenate(all_sar, axis=0)
        arr_opt = np.concatenate(all_opt, axis=0)

        l = len(arr)

        for i in tqdm(range(l)):
            image_out = Image.fromarray(arr[i])
            image_out.save(
                os.path.join(f'sample_results/{args.timestep_respacing}/gen_opt', f'{i + count * args.batch_size}.png'))
            image_sar = Image.fromarray(arr_sar[i])
            image_sar.save(os.path.join(f'sample_results/{args.timestep_respacing}/cond_sar',
                                        f'{i + count * args.batch_size}.png'))
            image_opt = Image.fromarray(arr_opt[i])
            image_opt.save(
                os.path.join(f'sample_results/{args.timestep_respacing}/gt_opt', f'{i + count * args.batch_size}.png'))
        count += 1
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {count * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]

    arr_sar = np.concatenate(all_sar, axis=0)
    arr_sar = arr_sar[: args.num_samples]

    arr_opt = np.concatenate(all_opt, axis=0)
    arr_opt = arr_opt[: args.num_samples]

    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        out_path_sar = os.path.join(logger.get_dir(), f"samples_sar_{shape_str}.npz")
        out_path_opt = os.path.join(logger.get_dir(), f"samples_opt_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
            np.savez(out_path_sar, arr_sar)
            np.savez(out_path_opt, arr_opt)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
