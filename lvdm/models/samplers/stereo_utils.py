from __future__ import annotations

import numpy as np
import torch
import torch.fft as fft
import torch.nn.functional as F
from lvdm.models.samplers.depth_utils import compute_uncertainty_mask


def _norm_depth(depth, max_val=1):
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = torch.zeros_like(depth)
    return out


def stereo_shift_torch_old(input_images, depthmaps, scale_factor=8, shift_both = False, stereo_offset_exponent=1.0):
    '''input: [C, T, H, W] depthmap: [T, H, W]'''

    def _create_stereo(input_images, depthmaps, scale_factor, stereo_offset_exponent):
        c, b, h, w = input_images.shape
        derived_image = torch.zeros_like(input_images)
        scale_factor_px = (scale_factor / 100.0) * input_images.shape[-1]

        for batch in range(b):
            for row in range(h):
                # Swipe order should ensure that pixels that are closer overwrite
                # (at their destination) pixels that are less close
                for col in range(w) if scale_factor_px < 0 else range(w - 1, -1, -1):
                    col_d = col + int((depthmaps[batch, row, col] ** stereo_offset_exponent) * scale_factor_px)
                    if 0 <= col_d < w:
                        derived_image[:, batch, row, col_d] = input_images[:, batch, row, col]

        return derived_image

    depthmaps = _norm_depth(depthmaps)

    if shift_both is False:
        left = input_images
        balance = 0
    else:
        balance = 0.5
        left = _create_stereo(input_images, depthmaps, + 1 * scale_factor * balance, stereo_offset_exponent)

    right = _create_stereo(input_images, depthmaps, - 1 * scale_factor * (1 - balance), stereo_offset_exponent)

    return torch.stack([left, right], dim=0)


def stereo_shift_torch(
    input_images, depthmaps, scale_factor=8, shift_both = False, stereo_offset_exponent=1.0
):
    """
    Aligns the left and right stereo views using the provided disparity map.

    Args:
        input_images: Tensor of shape (C, T, H, W) representing the left view.
        disparity_map: Tensor of shape (T, H, W) representing the disparity map between the views.
        scale_factor: A factor to control the amount of warping (default: 1.0).

    Returns:
        aligned_left, aligned_right: Aligned left and right view tensors.
    """

    def _create_stereo(input_images, depthmaps, scale_factor, stereo_offset_exponent):
        b, c, h, w = input_images.shape
        device = input_images.device

        # Initialize output tensors
        derived_image = torch.zeros_like(input_images)
        mask_image = torch.zeros_like(input_images)
        filling_image = torch.ones_like(input_images)

        # Compute scale factor in pixels
        scale_factor_px = (scale_factor / 100.0) * w
        print("scale_factor_px", scale_factor_px)

        # Generate index grids
        batch_indices = torch.arange(b, device=device).view(b, 1, 1).expand(b, h, w)
        row_indices = torch.arange(h, device=device).view(1, h, 1).expand(b, h, w)
        if scale_factor_px >= 0:
            col_order = torch.arange(w - 1, -1, -1, device=device)
        else:
            col_order = torch.arange(w, device=device)
        col_indices = col_order.view(1, 1, w).expand(b, h, w)

        # Extract depth values according to col_order
        depthmaps_ordered = depthmaps[:, :, :, col_order]

        # Compute the offset for each pixel
        offset = (depthmaps_ordered[:, 0, :, :] ** stereo_offset_exponent) * scale_factor_px
        offset = offset.long()

        # Calculate destination column indices
        col_d = col_indices + offset

        # Create a mask for valid destination indices within image boundaries
        mask = (col_d >= 0) & (col_d < w)

        # Filter indices where the mask is True
        valid_batch_inds = batch_indices[mask]
        valid_row_inds = row_indices[mask]
        valid_col_inds = col_indices[mask]
        valid_col_d = col_d[mask]

        # Reverse the indices to prioritize the last assignments
        # rev_valid_batch_inds = valid_batch_inds.flip(0)
        # rev_valid_row_inds = valid_row_inds.flip(0)
        # rev_valid_col_inds = valid_col_inds.flip(0)
        # rev_valid_col_d = valid_col_d.flip(0)
        rev_valid_batch_inds = valid_batch_inds.flip(0)
        rev_valid_row_inds = valid_row_inds.flip(0)
        rev_valid_col_inds = valid_col_inds.flip(0)
        rev_valid_col_d = valid_col_d.flip(0)

        # Compute destination indices
        dest_ids = rev_valid_batch_inds * (h * w) + rev_valid_row_inds * w + rev_valid_col_d

        # Convert dest_ids to NumPy array
        dest_ids_np = dest_ids.cpu().numpy()
        # Use NumPy to get unique indices
        _, unique_indices_np = np.unique(dest_ids_np, return_index=True)
        unique_indices = torch.from_numpy(unique_indices_np).to(device)

        # Extract the final indices for assignments
        final_batch_inds = rev_valid_batch_inds[unique_indices]
        final_row_inds = rev_valid_row_inds[unique_indices]
        final_col_inds = rev_valid_col_inds[unique_indices]
        final_col_d = rev_valid_col_d[unique_indices]

        # Perform the assignments using advanced indexing
        derived_image[final_batch_inds, :, final_row_inds, final_col_d] = input_images[final_batch_inds, :, final_row_inds, final_col_inds]
        mask_image[final_batch_inds, :, final_row_inds, final_col_d] = filling_image[final_batch_inds, :, final_row_inds, final_col_inds]

        return derived_image

    depthmaps = depthmaps[:, None]
    # depthmaps = _norm_depth(depthmaps)

    # (C, T, H, W) to (T, C, H, W)
    input_images = input_images.permute(1, 0, 2, 3)

    if not shift_both:
        left = input_images
        balance = 0
    else:
        balance = 0.5
        left = _create_stereo(input_images, depthmaps, + 1 * scale_factor * balance, stereo_offset_exponent)

    right = _create_stereo(input_images, depthmaps, - 1 * scale_factor * (1 - balance), stereo_offset_exponent)

    # (2, T, C, H, W) to (2, C, T, H, W)
    return torch.stack([left, right], dim=0).permute(0, 2, 1, 3, 4)


def align_stereo_views(
    left_view, right_view, disparity_map, stereo_offset_exponent=1.0, align_strength=0., threshold=0.1,
    mask_area_only=True
):
    """
    Aligns the left and right stereo views using the provided disparity map.

    Args:
        left_view: Tensor of shape (B, C, H, W) representing the left view.
        right_view: Tensor of shape (B, C, H, W) representing the right view.
        disparity_map: Tensor of shape (B, 1, H, W) representing the disparity map between the views.
        stereo_offset_exponent: A factor to control the amount of warping (default: 1.0).

    Returns:
        aligned_left, aligned_right: Aligned left and right view tensors.
    """

    disparity_map = _norm_depth(disparity_map)
    # B, C, H, W = left_view.shape
    # # Create a meshgrid for warping
    # grid_y, grid_x = torch.meshgrid(torch.arange(H, device=left_view.device), torch.arange(W, device=left_view.device))
    # grid_x = grid_x.unsqueeze(0).unsqueeze(0).float().repeat(B, 1, 1, 1)  # (1, 1, H, W)
    # grid_y = grid_y.unsqueeze(0).unsqueeze(0).float().repeat(B, 1, 1, 1)  # (1, 1, H, W)

    # scale_factor_px = (stereo_offset_exponent / 100.0) * W

    # # Warp the right view using disparity map
    # warped_grid_x = grid_x - scale_factor_px * (disparity_map)  # Apply disparity to x-coordinate
    # # Warp the right view using disparity map
    # # warped_grid_x = grid_x + stereo_offset_exponent * disparity_map  # Apply disparity to x-coordinate
    # warped_grid_x = 2.0 * warped_grid_x / (W - 1) - 1.0  # Normalize to [-1, 1] range for grid_sample
    # warped_grid_y = 2.0 * grid_y / (H - 1) - 1.0

    # # Combine x and y into a grid for warping
    # warped_grid = torch.stack((warped_grid_x, warped_grid_y), dim=-1).squeeze(1)  # (B, H, W, 2)

    # # Apply warping to the right view using grid_sample
    # left_aligned = F.grid_sample(
    #     left_view.clone(), warped_grid, mode='nearest', padding_mode='zeros', align_corners=False)
    left_warped, left_mask = create_stereo(left_view.clone(), disparity_map, stereo_offset_exponent, 1.)
    # left_mask = 1 - left_mask

    if mask_area_only:
        # left_mask = _create_stereo(torch.ones_like(left_view), disparity_map, stereo_offset_exponent, 1.)
        # left_mask = F.grid_sample(
        #     torch.ones_like(left_view), warped_grid, mode='nearest', padding_mode='zeros', align_corners=False)
        # uncertainty_mask = compute_uncertainty_mask(disparity_map, threshold).bool().repeat(1, 4, 1, 1)
        # left_mask[uncertainty_mask] = 0
        aligned_right = (left_warped * align_strength + right_view * (1 - align_strength)) * left_mask + right_view * (1 - left_mask)
    else:
        aligned_right = left_warped * align_strength + right_view * (1 - align_strength)

    return left_view, aligned_right


def warp_fft_low_frequency(noise, disparity, stereo_scale_factor):

    noise_freq = fft.fftn(noise, dim=(-2, -1))
    # mask_low, mask_high = create_frequency_masks(noise.shape[-2:], cutoff=0.05, device=noise.device) 
    # print(0.7, mask_high.sum(), mask_low.sum())
    # mask_low, mask_high = create_frequency_masks(noise.shape[-2:], cutoff=0.6, device=noise.device) 
    # print(0.6, mask_high.sum(), mask_low.sum())
    mask_low, mask_high = create_frequency_masks(noise.shape[-2:], cutoff=0.2, device=noise.device)
    # print(0.5, mask_high.sum(), mask_low.sum())  # Define low/high frequency masks
    noise_freq_low = noise_freq * mask_low
    noise_freq_high = noise_freq * mask_high
    noise_low = fft.ifftn(noise_freq_low, dim=(-2, -1)).real
    noise_high = fft.ifftn(noise_freq_high, dim=(-2, -1)).real

    disparity = torch.nn.functional.interpolate(
        disparity[0],
        size=[noise_low.size(-2), noise_low.size(-1)],
        mode="bicubic",
        align_corners=False
    )[None]
    noise_low_warped = stereo_shift_torch(
        noise_low[0], disparity[0, :, 0], scale_factor=stereo_scale_factor, shift_both=False
    )

    # Combine the warped low-frequency noise with the high-frequency component
    noise_combined = noise_low_warped + noise_high
    return noise_combined


def create_frequency_masks(size, cutoff=0.2, device="cpu"):
    """
    Create low- and high-frequency masks for separating Fourier components.
    
    Args:
        size: Size of the spatial dimensions (H, W).
        cutoff: Fraction of the frequency band for the low-frequency mask.
    
    Returns:
        mask_low, mask_high: Low- and high-frequency masks.
    """
    H, W = size
    mask_high = torch.zeros((H, W), dtype=torch.float32, device=device)
    mask_low = torch.ones((H, W), dtype=torch.float32, device=device)
    
    # Define the cutoff radius for low frequency (centered)
    center_x, center_y = H // 2, W // 2
    radius = int(min(H, W) * cutoff)
    
    for i in range(H):
        for j in range(W):
            distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
            if distance <= radius:
                mask_low[i, j] = 0.0
                mask_high[i, j] = 1.0
    
    # Expand to match input batch size and channels for broadcasting
    mask_low = mask_low[None, None, :, :]
    mask_high = mask_high[None, None, :, :]
    
    return mask_low, mask_high


def create_stereo(input_images, depthmaps, scale_factor, stereo_offset_exponent=1.):
    b, c, h, w = input_images.shape
    device = input_images.device

    # Initialize output tensors
    derived_image = torch.zeros_like(input_images)
    mask_image = torch.zeros_like(input_images)
    filling_image = torch.ones_like(input_images)

    # Compute scale factor in pixels
    scale_factor_px = (scale_factor / 100.0) * w

    # Generate index grids
    batch_indices = torch.arange(b, device=device).view(b, 1, 1).expand(b, h, w)
    row_indices = torch.arange(h, device=device).view(1, h, 1).expand(b, h, w)
    if scale_factor_px >= 0:
        col_order = torch.arange(w - 1, -1, -1, device=device)
    else:
        col_order = torch.arange(w, device=device)
    col_indices = col_order.view(1, 1, w).expand(b, h, w)

    # Extract depth values according to col_order
    depthmaps_ordered = depthmaps[:, :, :, col_order]

    # Compute the offset for each pixel
    offset = (depthmaps_ordered[:, 0, :, :] ** stereo_offset_exponent) * scale_factor_px
    offset = offset.long()

    # Calculate destination column indices
    col_d = col_indices + offset

    # Create a mask for valid destination indices within image boundaries
    mask = (col_d >= 0) & (col_d < w)

    # Filter indices where the mask is True
    valid_batch_inds = batch_indices[mask]
    valid_row_inds = row_indices[mask]
    valid_col_inds = col_indices[mask]
    valid_col_d = col_d[mask]

    # Reverse the indices to prioritize the last assignments
    rev_valid_batch_inds = valid_batch_inds.flip(0)
    rev_valid_row_inds = valid_row_inds.flip(0)
    rev_valid_col_inds = valid_col_inds.flip(0)
    rev_valid_col_d = valid_col_d.flip(0)

    # Compute destination indices
    dest_ids = rev_valid_batch_inds * (h * w) + rev_valid_row_inds * w + rev_valid_col_d

    # Convert dest_ids to NumPy array
    dest_ids_np = dest_ids.cpu().numpy()
    # Use NumPy to get unique indices
    _, unique_indices_np = np.unique(dest_ids_np, return_index=True)
    unique_indices = torch.from_numpy(unique_indices_np).to(device)

    # Extract the final indices for assignments
    final_batch_inds = rev_valid_batch_inds[unique_indices]
    final_row_inds = rev_valid_row_inds[unique_indices]
    final_col_inds = rev_valid_col_inds[unique_indices]
    final_col_d = rev_valid_col_d[unique_indices]

    # Perform the assignments using advanced indexing
    derived_image[final_batch_inds, :, final_row_inds, final_col_d] = input_images[final_batch_inds, :, final_row_inds, final_col_inds]
    mask_image[final_batch_inds, :, final_row_inds, final_col_d] = filling_image[final_batch_inds, :, final_row_inds, final_col_inds]

    return derived_image, mask_image