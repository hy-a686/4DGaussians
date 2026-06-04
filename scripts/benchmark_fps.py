import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mmcv
import numpy as np
import torch

from arguments import ModelHiddenParams, ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state
from utils.params_utils import merge_hparams


def render_views(views, gaussians, pipeline, background, cam_type):
    for view in views:
        render(view, gaussians, pipeline, background, stage="fine", cam_type=cam_type)


def main():
    parser = ArgumentParser(description="Benchmark render-only FPS after GPU warmup.")
    model = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--configs", required=True, type=str)
    parser.add_argument("--warmup", default=20, type=int)
    parser.add_argument("--repeats", default=5, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    args = merge_hparams(args, mmcv.Config.fromfile(args.configs))
    safe_state(args.quiet)

    dataset = model.extract(args)
    pipeline = pipeline_params.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, hyperparam.extract(args))
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    if pipeline.static_dynamic_routing:
        if not pipeline.routing_table_path:
            raise ValueError("--routing_table_path is required with --static_dynamic_routing")
        gaussians.load_deformation_table(pipeline.routing_table_path)

    test_cameras = scene.getTestCameras()
    views = [test_cameras[index] for index in range(len(test_cameras))]
    background_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(background_color, dtype=torch.float32, device="cuda")
    warmup_views = [views[index % len(views)] for index in range(args.warmup)]

    with torch.no_grad():
        render_views(warmup_views, gaussians, pipeline, background, scene.dataset_type)
        torch.cuda.synchronize()
        fps_results = []
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            render_views(views, gaussians, pipeline, background, scene.dataset_type)
            end.record()
            torch.cuda.synchronize()
            elapsed_seconds = start.elapsed_time(end) / 1000.0
            fps_results.append(len(views) / elapsed_seconds)

    dynamic_ratio = gaussians._deformation_table.float().mean().item()
    print("Gaussians:", gaussians.get_xyz.shape[0])
    print("dynamic ratio:", dynamic_ratio)
    print("FPS runs:", fps_results)
    print("FPS mean/std:", float(np.mean(fps_results)), float(np.std(fps_results)))


if __name__ == "__main__":
    main()
