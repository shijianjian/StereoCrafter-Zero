import argparse
import numpy as np
import os
import torch

from Video_Depth_Anything.video_depth_anything.video_depth import VideoDepthAnything
from Video_Depth_Anything.utils.dc_utils import read_video_frames, save_video


class VideoDepthAnythingWrapper():

    def __init__(
        self,
        encoder: str = "vitl",
        model_path = "/ibex/ai/home/shij0c/git/makeit3d/DynamiCrafter/Video_Depth_Anything",
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        self.model = VideoDepthAnything(**model_configs[encoder])
        self.model.load_state_dict(torch.load(f'{model_path}/checkpoints/video_depth_anything_{encoder}.pth', map_location='cpu'), strict=True)
        self.model = self.model.to(self.device).eval()

    def infer_frames(
        self,
        frames: torch.Tensor | np.ndarray,
        input_size=518,
        save_to: str | None = None,
        target_fps: int = 10,
        fp32: bool = False,
    ):
        depths, fps = self.model.infer_video_depth(frames, target_fps, input_size=input_size, device=self.device, fp32=fp32)
    
        if save_to:
            save_video(depths, save_to.replace(".mp4", "_depth.mp4"), fps=fps, is_depths=True, grayscale=True)
            save_video(depths, save_to.replace(".mp4", "_vis.mp4"), fps=fps, is_depths=True, grayscale=False)

        d_min, d_max = depths.min(), depths.max()

        return (depths - d_min) / (d_max - d_min)


if __name__ == "__main__":
    model = VideoDepthAnythingWrapper()
    print(model.infer_frames((np.random.rand(16, 512, 512, 3) * 255).astype(np.uint8), save_to="tmp/xxx.mp4").shape)
