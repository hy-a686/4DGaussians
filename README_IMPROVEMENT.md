# 4DGaussians Improvement Branch

This repository is forked from the official 4DGaussians codebase:

- Upstream paper: 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering, CVPR 2024
- Upstream repository: https://github.com/hustvl/4DGaussians
- Working branch for this course project: `improvement`

This branch contains the code used in the SLAM final project report. The report sections are mapped as follows:

- Section 4: baseline reproduction
- Section 5: identified limitations
- Section 6.1: H1, motion-aware deformation regularization
- Section 6.2: H2, Static/Dynamic Branch Selection Based on Deformation Magnitude
- Section 7: comparison experiments and ablations

## What Was Changed

### H1: motion-aware deformation regularization

H1 adds a training-time regularization term for the deformation field. The motivation is that dynamic scenes contain frames and regions with uneven motion strength. A single deformation representation may either underfit large motion or disturb weak-motion regions. H1 therefore computes a motion response from the predicted deformation and applies an additional loss to emphasize high-motion samples.

Main files:

- `utils/motion_utils.py`
- `train.py`
- `utils/loss_utils.py`
- `arguments/__init__.py`
- `arguments/dnerf/hook_motion_v2.py`
- `arguments/hypernerf/broom2_motion_v2.py`

Important configuration keys:

- `motion_loss_weight`
- `motion_loss_quantile`
- `motion_loss_min_response`
- `motion_loss_group_by_stream`

### H2: Static/Dynamic Branch Selection Based on Deformation Magnitude

H2 is an inference-time acceleration attempt. A Gaussian-level deformation table is built by sampling the learned deformation over time. Gaussians with deformation magnitude below a threshold are treated as static candidates, while the remaining Gaussians keep the dynamic deformation branch. This reduces unnecessary deformation computation for near-static points.

The command-line flag is still named `--static_dynamic_routing` in code for compatibility with earlier experiments, but it corresponds to the branch selection method described in the report.

Main files:

- `scene/gaussian_model.py`
- `gaussian_renderer/__init__.py`
- `render.py`
- `scripts/build_deformation_table.py`
- `scripts/benchmark_fps.py`

## Environment

The original repository recommends PyTorch 1.13.1 with CUDA 11.6. My reproduction was completed on a Windows CUDA 11.8 environment because of local driver and toolchain constraints.

Tested local environment:

- OS: Windows 11
- Python: 3.10
- CUDA runtime: 11.8
- PyTorch: 2.0.1+cu118
- TorchVision: 0.15.2+cu118
- Main Python dependencies: `mmcv==1.6.0`, `lpips`, `plyfile`, `pytorch-msssim`, `open3d`, `imageio[ffmpeg]`, `ninja`

Suggested setup:

```powershell
conda create -n 4dgs python=3.10 -y
conda activate 4dgs
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install mmcv==1.6.0 matplotlib argparse lpips plyfile pytorch_msssim open3d imageio[ffmpeg] ninja
python -m pip install -e submodules/depth-diff-gaussian-rasterization --no-build-isolation
python -m pip install -e submodules/simple-knn --no-build-isolation
```

If the official PyTorch 1.13.1+cu116 environment is available, it can also be used with the original `requirements.txt`. The CUDA extension submodules must be compiled in the same Python environment that runs training.

## Dataset Layout

The commands below assume the following dataset layout:

```text
data/hook
data/hypernerf/virg/broom2
```

Large datasets and rendered outputs are not intended to be committed to GitHub. They should be submitted in the experiment data ZIP required by the assignment.

## Baseline Reproduction Commands

### Synthetic scene: d-NeRF Hook

```powershell
python train.py -s data/hook --port 6017 --expname "baseline/dnerf/hook" --configs arguments/dnerf/hook.py
python render.py -m output/baseline/dnerf/hook -s data/hook --configs arguments/dnerf/hook.py --iteration 20000 --skip_train --skip_video
python metrics.py -m output/baseline/dnerf/hook
```

### Real scene: HyperNeRF Broom2

```powershell
python train.py -s data/hypernerf/virg/broom2 --port 6017 --expname "baseline/hypernerf/broom2" --configs arguments/hypernerf/broom2.py
python render.py -m output/baseline/hypernerf/broom2 -s data/hypernerf/virg/broom2 --configs arguments/hypernerf/broom2.py --iteration 14000 --skip_train --skip_video
python metrics.py -m output/baseline/hypernerf/broom2
```

## Improved Method Commands

### H1 only: motion-aware deformation regularization

Synthetic Hook:

```powershell
python train.py -s data/hook --port 6021 --expname "improved/h1_v2/hook" --configs arguments/dnerf/hook_motion_v2.py
python render.py -m output/improved/h1_v2/hook -s data/hook --configs arguments/dnerf/hook_motion_v2.py --iteration 20000 --skip_train --skip_video
python metrics.py -m output/improved/h1_v2/hook
```

Real Broom2:

```powershell
python train.py -s data/hypernerf/virg/broom2 --port 6019 --expname "improved/h1_v2/broom2" --configs arguments/hypernerf/broom2_motion_v2.py
python render.py -m output/improved/h1_v2/broom2 -s data/hypernerf/virg/broom2 --configs arguments/hypernerf/broom2_motion_v2.py --iteration 14000 --skip_train --skip_video
python metrics.py -m output/improved/h1_v2/broom2
```

### H2 only: static/dynamic branch selection from the baseline model

The following one-line PowerShell commands prepare a copied model directory, build the deformation table, render the test set, compute metrics, and benchmark render-only FPS. The threshold used in the final comparison is `0.5`; other thresholds were used for ablation.

Synthetic Hook:

```powershell
$baseline="output/baseline/dnerf/hook"; $dest="output/improved/h2/hook_t0.5"; New-Item -ItemType Directory -Force "$dest/point_cloud" | Out-Null; Copy-Item "$baseline/cfg_args" "$dest/cfg_args" -Force; Copy-Item "$baseline/point_cloud/iteration_20000" "$dest/point_cloud/iteration_20000" -Recurse -Force; python scripts/build_deformation_table.py -m $baseline -s data/hook --configs arguments/dnerf/hook.py --iteration 20000 --threshold 0.5 --samples 25 --output "$dest/deformation_table.pth"; python render.py -m $dest -s data/hook --configs arguments/dnerf/hook.py --iteration 20000 --skip_train --skip_video --static_dynamic_routing --routing_table_path "$dest/deformation_table.pth"; python metrics.py -m $dest; python scripts/benchmark_fps.py -m $dest -s data/hook --configs arguments/dnerf/hook.py --iteration 20000 --static_dynamic_routing --routing_table_path "$dest/deformation_table.pth"
```

Real Broom2:

```powershell
$baseline="output/baseline/hypernerf/broom2"; $dest="output/improved/h2/broom2_t0.5"; New-Item -ItemType Directory -Force "$dest/point_cloud" | Out-Null; Copy-Item "$baseline/cfg_args" "$dest/cfg_args" -Force; Copy-Item "$baseline/point_cloud/iteration_14000" "$dest/point_cloud/iteration_14000" -Recurse -Force; python scripts/build_deformation_table.py -m $baseline -s data/hypernerf/virg/broom2 --configs arguments/hypernerf/broom2.py --iteration 14000 --threshold 0.5 --samples 25 --output "$dest/deformation_table.pth"; python render.py -m $dest -s data/hypernerf/virg/broom2 --configs arguments/hypernerf/broom2.py --iteration 14000 --skip_train --skip_video --static_dynamic_routing --routing_table_path "$dest/deformation_table.pth"; python metrics.py -m $dest; python scripts/benchmark_fps.py -m $dest -s data/hypernerf/virg/broom2 --configs arguments/hypernerf/broom2.py --iteration 14000 --static_dynamic_routing --routing_table_path "$dest/deformation_table.pth"
```

### H1 + H2 combined

The combined experiment first trains H1, then builds the H2 deformation table from the H1 checkpoint.

Synthetic Hook:

```powershell
$baseline="output/improved/h1_v2/hook"; $dest="output/improved/h1_h2/hook_t0.5"; New-Item -ItemType Directory -Force "$dest/point_cloud" | Out-Null; Copy-Item "$baseline/cfg_args" "$dest/cfg_args" -Force; Copy-Item "$baseline/point_cloud/iteration_20000" "$dest/point_cloud/iteration_20000" -Recurse -Force; python scripts/build_deformation_table.py -m $baseline -s data/hook --configs arguments/dnerf/hook_motion_v2.py --iteration 20000 --threshold 0.5 --samples 25 --output "$dest/deformation_table.pth"; python render.py -m $dest -s data/hook --configs arguments/dnerf/hook_motion_v2.py --iteration 20000 --skip_train --skip_video --static_dynamic_routing --routing_table_path "$dest/deformation_table.pth"; python metrics.py -m $dest; python scripts/benchmark_fps.py -m $dest -s data/hook --configs arguments/dnerf/hook_motion_v2.py --iteration 20000 --static_dynamic_routing --routing_table_path "$dest/deformation_table.pth"
```

Real Broom2:

```powershell
$baseline="output/improved/h1_v2/broom2"; $dest="output/improved/h1_h2/broom2_t0.5"; New-Item -ItemType Directory -Force "$dest/point_cloud" | Out-Null; Copy-Item "$baseline/cfg_args" "$dest/cfg_args" -Force; Copy-Item "$baseline/point_cloud/iteration_14000" "$dest/point_cloud/iteration_14000" -Recurse -Force; python scripts/build_deformation_table.py -m $baseline -s data/hypernerf/virg/broom2 --configs arguments/hypernerf/broom2_motion_v2.py --iteration 14000 --threshold 0.5 --samples 25 --output "$dest/deformation_table.pth"; python render.py -m $dest -s data/hypernerf/virg/broom2 --configs arguments/hypernerf/broom2_motion_v2.py --iteration 14000 --skip_train --skip_video --static_dynamic_routing --routing_table_path "$dest/deformation_table.pth"; python metrics.py -m $dest; python scripts/benchmark_fps.py -m $dest -s data/hypernerf/virg/broom2 --configs arguments/hypernerf/broom2_motion_v2.py --iteration 14000 --static_dynamic_routing --routing_table_path "$dest/deformation_table.pth"
```

## H2 Ablation Commands

For Broom2, the ablation thresholds used in the report were `0.25`, `0.5`, `1`, `5`, and `10`. Replace `$t` in the command below:

```powershell
$t=0.5; $baseline="output/baseline/hypernerf/broom2"; $dest="output/improved/h2/broom2_t$t"; New-Item -ItemType Directory -Force "$dest/point_cloud" | Out-Null; Copy-Item "$baseline/cfg_args" "$dest/cfg_args" -Force; Copy-Item "$baseline/point_cloud/iteration_14000" "$dest/point_cloud/iteration_14000" -Recurse -Force; python scripts/build_deformation_table.py -m $baseline -s data/hypernerf/virg/broom2 --configs arguments/hypernerf/broom2.py --iteration 14000 --threshold $t --samples 25 --output "$dest/deformation_table.pth"; python render.py -m $dest -s data/hypernerf/virg/broom2 --configs arguments/hypernerf/broom2.py --iteration 14000 --skip_train --skip_video --static_dynamic_routing --routing_table_path "$dest/deformation_table.pth"; python metrics.py -m $dest; python scripts/benchmark_fps.py -m $dest -s data/hypernerf/virg/broom2 --configs arguments/hypernerf/broom2.py --iteration 14000 --static_dynamic_routing --routing_table_path "$dest/deformation_table.pth"
```

## Experiment Logs

The local logs used by the report are stored in `logs/`. Key files include:

- Baseline: `logs/baseline_hook.log`, `logs/render_hook.log`, `logs/metrics_hook.log`, `logs/baseline_broom2.log`, `logs/render_broom2.log`, `logs/metrics_broom2.log`
- H1: `logs/h1_v2_hook.log`, `logs/render_h1_v2_hook.log`, `logs/metrics_h1_v2_hook.log`, `logs/h1_v2_broom2.log`, `logs/render_h1_v2_broom2.log`, `logs/metrics_h1_v2_broom2.log`
- H2: `logs/h2_build_table_hook_t0.5.log`, `logs/h2_render_hook_t0.5.log`, `logs/h2_metrics_hook_t0.5.log`, `logs/h2_fps_hook_t0.5.log`, `logs/h2_build_table_t0.5.log`, `logs/h2_render_t0.5.log`, `logs/h2_metrics_t0.5.log`, `logs/h2_fps_t0.5.log`
- H1 + H2: `logs/h1_h2_build_hook.log`, `logs/h1_h2_render_hook.log`, `logs/h1_h2_metrics_hook.log`, `logs/h1_h2_fps_hook.log`, `logs/h1_h2_build_broom2.log`, `logs/h1_h2_render_broom2.log`, `logs/h1_h2_metrics_broom2.log`, `logs/h1_h2_fps_broom2.log`

Metrics are written to `results.json` and `per_view.json` inside each output directory after running `metrics.py`. Rendered images and metric JSON files should be included in the assignment ZIP rather than committed to this repository.

## Notes on Reproducibility

- Baseline and improved experiments should be run on the same machine, same dataset split, and same checkpoint iteration.
- H2 does not retrain the model. It reuses either the baseline checkpoint or the H1 checkpoint and changes only the inference branch selection.
- The `scripts/benchmark_fps.py` script measures render-only FPS after GPU warmup. This is different from the FPS printed by `render.py`, which includes file writing overhead and is therefore less stable.
- Random state initialization follows the original repository through `safe_state()`.

## Files Not Included in GitHub

The following artifacts are intentionally excluded from the GitHub repository and should be placed in the assignment experiment data ZIP if needed:

- datasets under `data/`
- downloaded archives under `downloads/`
- model outputs under `output/`
- zip archives such as `data.zip`
- rendered videos and temporary visualization files
