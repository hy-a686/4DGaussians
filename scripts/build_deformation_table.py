import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mmcv
import numpy as np
import torch

from arguments import ModelHiddenParams, ModelParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene
from utils.general_utils import safe_state
from utils.params_utils import merge_hparams


def main():
    parser = ArgumentParser(description="Build a static-dynamic Gaussian routing table.")
    model = ModelParams(parser, sentinel=True)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--configs", required=True, type=str)
    parser.add_argument("--samples", default=25, type=int)
    parser.add_argument("--threshold", default=0.01, type=float)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    args = merge_hparams(args, mmcv.Config.fromfile(args.configs))
    safe_state(args.quiet)

    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, hyperparam.extract(args))
    Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    time_samples = np.linspace(0.0, 1.0, args.samples)
    score = gaussians.build_deformation_table(time_samples, args.threshold)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(gaussians._deformation_table.cpu(), args.output)
    dynamic_ratio = gaussians._deformation_table.float().mean().item()
    print("routing table:", args.output)
    print("Gaussians:", score.numel())
    print("dynamic ratio:", dynamic_ratio)
    print("score min/mean/max:", score.min().item(), score.mean().item(), score.max().item())
    quantiles = torch.quantile(score, torch.tensor([0.25, 0.5, 0.75, 0.9, 0.95, 0.99], device=score.device))
    print("score quantiles [25%, 50%, 75%, 90%, 95%, 99%]:", quantiles.cpu().tolist())


if __name__ == "__main__":
    main()
