import cv2
import numpy as np
import torch


def _camera_to_gray(camera):
    image = camera.original_image.detach().cpu().permute(1, 2, 0).numpy()
    image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _camera_stream_key(camera):
    stem = str(getattr(camera, "image_name", "")).rsplit(".", 1)[0]
    prefix, separator, suffix = stem.rpartition("_")
    if separator and suffix.isdigit():
        return prefix
    return "__default__"


def _flow_residual_map(source_gray, target_gray, quantile, min_response):
    flow = cv2.calcOpticalFlowFarneback(
        source_gray,
        target_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    global_flow = np.median(flow.reshape(-1, 2), axis=0)
    magnitude = np.linalg.norm(flow - global_flow, axis=-1)
    scale = max(float(np.quantile(magnitude, quantile)), 1e-6)
    motion_map = np.clip(magnitude / scale, 0.0, 1.0)
    if min_response > 0:
        if min_response >= 1:
            raise ValueError("motion_loss_min_response must be smaller than 1.")
        motion_map = np.clip((motion_map - min_response) / (1.0 - min_response), 0.0, 1.0)
    return motion_map.astype(np.float32)


class MotionWeightCache:
    """Precompute temporal motion maps outside the training loop."""

    def __init__(self, cameras, quantile=0.95, min_response=0.0, group_by_stream=False):
        camera_list = [cameras[index] for index in range(len(cameras))]
        streams = {"__all__": camera_list}
        if group_by_stream:
            streams = {}
            for camera in camera_list:
                streams.setdefault(_camera_stream_key(camera), []).append(camera)
        self.weights = {}

        for stream in streams.values():
            ordered = sorted(stream, key=lambda camera: (camera.time, camera.uid))
            gray_images = [_camera_to_gray(camera) for camera in ordered]
            
            for index, camera in enumerate(ordered):
                maps = []
                if index > 0:
                    maps.append(_flow_residual_map(gray_images[index], gray_images[index - 1], quantile, min_response))
                if index + 1 < len(ordered):
                    maps.append(_flow_residual_map(gray_images[index], gray_images[index + 1], quantile, min_response))

                if maps:
                    motion_map = np.maximum.reduce(maps)
                else:
                    motion_map = np.zeros_like(gray_images[index], dtype=np.float32)
                self.weights[camera.uid] = torch.from_numpy(motion_map).unsqueeze(0)

    def get(self, camera, device):
        return self.weights[camera.uid].to(device=device, non_blocking=True)