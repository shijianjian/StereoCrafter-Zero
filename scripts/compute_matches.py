import torchvision
import torch

from lvdm.models.samplers.stereo_utils import (
    stereo_shift_torch, align_stereo_views, warp_fft_low_frequency, create_stereo, stereo_shift_torch_old,
    _norm_depth,
)

def compute_depthcrafter(images, save_name=None):
    from depthcrafter import DepthCrafterWrapper
    # images = images.permute(0, 2, 3, 1)
    model = DepthCrafterWrapper(
        unet_path="tencent/DepthCrafter",
        pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
        cpu_offload="model",
    )
    return torch.tensor(
        model.infer_frames(images.cpu().numpy(), save_to=save_name), device=images.device, dtype=images.dtype
    )[:, None]


def get_videos(video_path, stereo=False):
    video, audio, info = torchvision.io.read_video(video_path, pts_unit="sec")

    # Get video dimensions
    T, H, W, C = video.shape  # (num_frames, height, width, channels)
    if not stereo:
        return video

    # Split into left and right parts
    mid = W // 2
    left_video = video[:, :, :mid, :]  # Left half
    right_video = video[:, :, mid:, :]  # Right half
    return left_video, right_video


def compute_matched_pixels(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor, tolerance: int = 5):
    """
    Compare two CBHW image tensors and compute the percentage of matched pixels within the given RGB tolerance.

    Args:
        tensor1 (torch.Tensor): First image tensor of shape (C, B, H, W).
        tensor2 (torch.Tensor): Second image tensor of shape (C, B, H, W).
        tolerance (int): Allowed difference per channel for a pixel to be considered a match.

    Returns:
        float: Percentage of matched pixels.
    """

    assert tensor1.shape == tensor2.shape, "Input tensors must have the same shape."
    assert tensor1.dim() == 4, "Tensors must have shape (C, B, H, W)."

    # Compute absolute difference and check if within tolerance for all channels
    matched_pixels = (torch.abs(tensor1 - tensor2) <= tolerance).all(dim=0)

    matched_pixels[~mask] = 0

    # Count the number of matched pixels
    num_matched_pixels = matched_pixels.sum().item()

    # Compute total number of pixels
    total_pixels = torch.ones_like(mask)[mask].sum().item()

    # Compute match percentage
    match_percentage = (num_matched_pixels / total_pixels) * 100

    return match_percentage


if __name__ == "__main__":
    scale_factor = 8
    # lv, rv = get_videos("/ibex/ai/home/shij0c/git/makeit3d/DynamiCrafter/tmp/a_smiling_girl-r.mp4")

    # lv, rv = get_videos("/ibex/ai/home/shij0c/git/makeit3d/DynamiCrafter/tmp_depthcrafter/a_beautiful_woman_with_long_hair_and_a_d.mp4", stereo=True)
    # rv = get_videos("/ibex/ai/home/shij0c/git/makeit3d/DynamiCrafter/ablations/ProPainter/results/a_beautiful_woman_with_long_hair_and_a_d_masked/inpaint_out.mp4")


    rv, lv = get_videos("/ibex/ai/home/shij0c/git/makeit3d/DynamiCrafter/tmp_512/a_woman_looking_out_in_the_rain-r.mp4", stereo=True)
    lv = get_videos("/ibex/ai/home/shij0c/git/makeit3d/DynamiCrafter/ablations/ProPainter/results/a_woman_looking_out_in_the_rain_masked/inpaint_out.mp4")

    disparity = compute_depthcrafter(rv / 255., "xx.mp4")
    print("disparity", disparity.min(), disparity.max())
    t, h, w, c = lv.shape
    warpped = stereo_shift_torch(
        rv.permute(3, 0, 1, 2), disparity[:, 0], scale_factor=scale_factor, shift_both=False)
    warpped_l = warpped[:1]
    warpped_r = warpped[1:]
    torchvision.io.write_video(
        "aa.mp4", warpped_r[0].permute(1, 2, 3, 0), fps=10, video_codec='h264', options={'crf': '10'})
    torchvision.io.write_video(
        "ab.mp4", warpped_l[0].permute(1, 2, 3, 0), fps=10, video_codec='h264', options={'crf': '10'})
    mask_r = warpped_r.reshape(-1, c, t, h, w)[:, 0, :, ...] != 0

    lv = lv.permute(3, 0, 1, 2)

    res = compute_matched_pixels(lv, warpped_r[0].cpu(), mask_r[0], 35)
    print(res)
    # err = (lv - warpped_r[0].cpu())
    # print(lv.shape, warpped_r[0].cpu().shape, err.shape)
    # err = err[mask_r[0]].float()
    # print(lv.shape, warpped_r[0].shape, err.shape, mask_r.shape, err.mean())

    # print(lv.shape, rv.shape, warpped_l.shape, warpped_r.shape,  mask_r.shape, err.mean())

