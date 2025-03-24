from __future__ import annotations
from typing import Any
import numpy as np
from tqdm import tqdm
import torch
from lvdm.models.utils_diffusion import make_ddim_sampling_parameters, make_ddim_timesteps, rescale_noise_cfg
from lvdm.common import noise_like
from lvdm.common import extract_into_tensor
from lvdm.models.samplers.ddim import DDIMSampler
from einops import rearrange
import copy
import kornia
import torchvision.transforms as transforms
import torch.nn.functional as F
from diffusers import AutoPipelineForInpainting

from lvdm.models.samplers.depth_utils import compute_uncertainty_mask


import torch
from scipy import spatial


class Inpainting:
    """
    Inpaints a tensor using Dirichlet interpolation.
    Works on tensors of shape (C, H, W) and a binary mask of shape (1, H, W).

    Modified from: https://github.com/aGIToz/PyInpaint/tree/main
    """

    def __init__(self, ps):
        """
        ps: patch size (used for creating a dynamic non-local graph)
        """
        self.ps = ps

    def preprocess(self, latents, mask, guidance=None):
        """
        Preprocess the latents and mask to set up for inpainting.
        latents: input tensor of shape (C, H, W)
        mask: binary mask tensor of shape (1, H, W)
        """
        # Ensure latents and mask have the correct shape and type
        latents = latents.float()
        mask = mask.float()

        # Apply mask to latents (masked regions will be zero)
        latents = latents * mask

        # Store shape
        self._shape = latents.shape

        # Use guidance if provided, otherwise the latent itself is used for inpainting
        if guidance is not None:
            self.guidance = guidance.float() * (1 - mask)  # Only use guidance where the mask is zero
        else:
            self.guidance = latents.clone()  # Default to using latent features as guidance

        # Generate position feature matrix
        self._position = pmat(self._shape)

        # Flatten the latents tensor into a texture (spatial dimensions only)
        self._texture = latents.view(latents.size(0), -1).T  # Shape (H*W, C)

        # Create patches from the image/tensor
        self._patches = create_patches(latents, (self.ps, self.ps))

    def postprocess(self, fmat):
        """
        Reshape the flattened feature map back to the original latent dimensions.
        """
        return fmat.T.view(self._shape)

    def forward(self, latents, mask, guidance=None, guidance_weight=0.5, k_boundary=4, k_search=1000, k_patch=5):
        """
        Inpainting process to fill masked areas in the tensor.
        """
        self.preprocess(latents, mask, guidance)

        kdt = spatial.cKDTree(self._position.numpy())
        dA = torch.where(self._texture.any(dim=1))[0]
        A = torch.where(~self._texture.any(dim=1))[0]

        while A.size(0) >= 1:
            dmA = torch.empty(0, device=A.device, dtype=torch.long)

            for i in A:
                _, indices = kdt.query(self._position[i].numpy(), k_boundary)
                if (~torch.isin(torch.tensor(indices, device=A.device), A)).any():
                    dmA = torch.cat([dmA, i.unsqueeze(0)])
                    mask = (~(self._patches[i].flatten() == 0)).float()
                    _, indices = kdt.query(self._position[i].numpy(), k_search)
                    indices = torch.tensor(indices, device=A.device)
                    part_of_dA = indices[~torch.isin(indices, A)]
                    new_patches = mask.flatten() * self._patches[part_of_dA]
                    kdt_ = spatial.cKDTree(new_patches.cpu().numpy())
                    _, indices = kdt_.query(self._patches[i].flatten().cpu().numpy(), k_patch)
                    indices = torch.tensor(indices, device=A.device)
                    ids = part_of_dA[indices]

                    if guidance is not None:
                        # Blend the texture from guidance with the surrounding patches
                        self._texture[i] = guidance_weight * self.guidance.view(self._shape[0], -1)[:, i] + \
                                        (1 - guidance_weight) * self._texture[ids].mean(dim=0)
                    else:
                        self._texture[i] = self._texture[ids].mean(dim=0)

            self._patches = create_patches(self._texture.reshape(self._shape), (self.ps, self.ps))
            dA = torch.cat([dA, dmA])
            A = A[~torch.isin(A, dmA)]

        return self.postprocess(self._texture)


# Utility functions for pmat and create_patches

def pmat(shape):
    """
    Returns the position feature matrix for a given tensor shape.
    Assumes the shape is (C, H, W) for tensors.
    """
    h, w = shape[1], shape[2]
    
    # Create meshgrid in PyTorch
    x = torch.arange(0, w).float()
    y = torch.arange(h, 0, -1).float()

    meshx, meshy = torch.meshgrid(x, y, indexing='xy')

    # Flatten the positions
    x = meshx.reshape(-1, 1)
    y = meshy.reshape(-1, 1)

    # Concatenate x and y positions, normalize by max dimension
    pmat = torch.cat((x, y), dim=1) / max(h, w)

    return pmat


def create_patches(img, patch_shape=(3, 3)):
    """
    Creates overlapping patches from the input tensor.
    img: Tensor of shape (C, H, W) or (H, W)
    patch_shape: Tuple indicating patch size (height, width).
    """
    if img.dim() == 2:  # Handle grayscale or 2D image (no channels)
        img = img.unsqueeze(0)  # Convert to (C=1, H, W)

    d, h, w = img.shape
    r, c = patch_shape

    # Padding

    pad_h = (int((r - 0.5) / 2.), int((r + 0.5) / 2.))
    pad_w = (int((c - 0.5) / 2.), int((c + 0.5) / 2.))

    img = F.pad(img, pad=(pad_w[0], pad_w[1], pad_h[0], pad_h[1]), mode='reflect')

    # Unfold the image into patches
    patches = img.permute(1, 2, 0).unfold(0, r, 1).unfold(1, c, 1)
    # Reshape to (num_patches, patch_size)
    patches = patches.contiguous().view(h * w, r * c * d)

    return patches


class DDIMStereoSampler(DDIMSampler):

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               schedule_verbose=False,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               precision=None,
               fs=None,
               timestep_spacing='uniform', #uniform_trailing for starting from last timestep
               guidance_rescale=0.0,
               is_duplicate=False,
               disparity=None,
               masked_latents=None,
               masked_latents_weight=0.7,
               is_final=True,
               cur_iter=None,
               stereo_repeat_times=1,
               total_iter=None,
               modify_step=None,
               step_and_latent=None,
               **kwargs
               ):
        
        # check condition bs
        if conditioning is not None:
            if isinstance(conditioning, dict):
                try:
                    cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                except:
                    cbs = conditioning[list(conditioning.keys())[0]][0].shape[0]

                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_discretize=timestep_spacing, ddim_eta=eta, verbose=schedule_verbose)

        # make shape
        if len(shape) == 3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        elif len(shape) == 4:
            C, T, H, W = shape
            size = (batch_size, C, T, H, W)

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    verbose=verbose,
                                                    precision=precision,
                                                    fs=fs,
                                                    guidance_rescale=guidance_rescale,
                                                    is_duplicate=is_duplicate,
                                                    disparity=disparity,
                                                    masked_latents=masked_latents,
                                                    masked_latents_weight=masked_latents_weight,
                                                    is_final=is_final,
                                                    cur_iter=cur_iter,
                                                    total_iter=total_iter,
                                                    stereo_repeat_times=stereo_repeat_times,
                                                    modify_step=modify_step,
                                                    step_and_latent=step_and_latent,
                                                    **kwargs)
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, verbose=True,precision=None,fs=None,guidance_rescale=0.0,
                      stereo_sacle_factor=8, masked_latents=None, masked_latents_weight=0.7, is_final=True, cur_iter=None, total_iter=None, modify_step=None,
                      fuse_latent=False, stereo_repeat_times=1, imtext_cond_warped=None, repeat_steps=None,
                      is_duplicate=False, disparity=None, step_and_latent=None, **kwargs):
        device = self.model.betas.device        
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if precision is not None:
            if precision == 16:
                img = img.to(dtype=torch.float16)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        if is_duplicate:
            img = img.repeat(2, *([1] * (len(shape) - 1)))
            b = b * 2
            if mask is not None:
                mask = mask.repeat(2, *([1] * (len(mask.shape) - 1)))
            fs = fs.repeat(2, *([1] * (len(fs.shape) - 1))) if fs is not None else fs

            if isinstance(cond, dict):
                cond_copy = copy.deepcopy(cond)
                for key in cond.keys():
                    if isinstance(cond[key], (list, tuple,)) and len(cond[key]) == 1:
                        cond_copy.update({key: [cond[key][0].repeat(2, *([1] * (len(cond[key][0].shape) - 1)))]})
                    elif isinstance(cond[key], torch.Tensor):
                        cond_copy.update({cond[key].repeat(2, *([1] * (len(cond[key].shape) - 1)))})
                    else:
                        raise RuntimeError
                cond = cond_copy
            else:
                cond = cond.repeat(2, *([1] * (len(cond.shape) - 1)))

            if isinstance(unconditional_conditioning, dict):
                uc = copy.deepcopy(unconditional_conditioning)
                for key in unconditional_conditioning.keys():
                    if isinstance(unconditional_conditioning[key], (list, tuple,)) and len(unconditional_conditioning[key]) == 1:
                        uc.update({key: [unconditional_conditioning[key][0].repeat(2, *([1] * (len(unconditional_conditioning[key][0].shape) - 1)))]})
                    elif isinstance(unconditional_conditioning[key], torch.Tensor):
                        uc.update({unconditional_conditioning[key].repeat(2, *([1] * (len(unconditional_conditioning[key].shape) - 1)))})
                    else:
                        raise RuntimeError
                unconditional_conditioning = uc
            else:
                unconditional_conditioning = unconditional_conditioning.repeat(2, *([1] * (len(unconditional_conditioning.shape) - 1)))

        intermediates = {'x_inter': [img], 'pred_x0': [img], "masked_latents": [], "masked_latents_left": [], "init_xT": [img], "mask_r": []}

        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        if verbose:
            iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        else:
            iterator = time_range

        clean_cond = kwargs.pop("clean_cond", False)

        # cond_copy, unconditional_conditioning_copy = copy.deepcopy(cond), copy.deepcopy(unconditional_conditioning)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if step_and_latent is not None:
                if i < step_and_latent[0] + 1:
                    print(f"skip step {i} / {step_and_latent[0]}")
                    continue
                if i == step_and_latent[0] + 1:
                    img = step_and_latent[1]
                    mask_r = step_and_latent[2]

            ## use mask to blend noised original latent (img_orig) & new sampled latent (img)
            if mask is not None:
                assert x0 is not None
                if clean_cond:
                    img_orig = x0
                else:
                    img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass? <ddim inversion>
                img = img_orig * mask + (1. - mask) * img # keep original & modify use img

            modify_steps = [] if modify_step is None else [modify_step]

            if repeat_steps is not None:
                modify_steps = [st for (st, rp_times) in repeat_steps]

            if i in modify_steps and repeat_steps is None:
                b, t, c, h, w = img.shape
                assert b == 2
                sacle_factor = stereo_sacle_factor
                shift_both = False
                disparity = torch.nn.functional.interpolate(
                    disparity[0],
                    size=[img.size(-2), img.size(-1)],
                    mode="bicubic",
                    align_corners=False
                )[None]
                latents = img
                latents_ts = stereo_shift_torch(
                    latents[0], disparity[0, :, 0], sacle_factor=sacle_factor, shift_both=shift_both)

                if shift_both:
                    latent_l = latents_ts[:1]  # Shape: (torch.Size([1, 4, 16, 72, 128]))
                    latent_r = latents_ts[1:]  # Shape: (torch.Size([1, 4, 16, 72, 128]))
                else:
                    latent_l = latents[:1]
                    latent_r = latents_ts[1:]

                make_latent_grid(latent_l, "latent_l")
                make_latent_grid(latent_r, "latent_r")

                mask_r = latent_r.reshape(-1, t, c, h, w)[:, 0, :, ...] != 0
                mask_r = rearrange(mask_r, 'b t h w -> b () t h w').repeat(1, 4, 1, 1, 1)
                mask_l = latent_l.reshape(-1, t, c, h, w)[:, 0, :, ...] != 0
                mask_l = rearrange(mask_l, 'b t h w -> b () t h w').repeat(1, 4, 1, 1, 1)

                # latent_r_src = latent_r.clone()
                # latent_l_src = latent_l.clone()

                # latent_l_src = kornia.filters.gaussian_blur2d(
                #     latent_r[0].clone(), (3, 3), (1, 1))[None].to(latent_r)
                # latent_l_src = kornia.filters.joint_bilateral_blur(
                #     latent_r[0].clone(), latent_l[0].clone(), (3, 3), 0.1, (1.5, 1.5))[None].to(latent_r)
                # print(latent_l.shape, latent_r.shape)
                # latent_l[~mask_l] = latent_r_src[~mask_l]

                filling_latents: torch.Tensor
                if masked_latents is None:
                    inpainting = Inpainting(ps=3)
                    filling_latents = torch.stack([
                        inpainting.forward(latent_r[0, :, j], mask_r[0, :, j], guidance=latent_l[0, :, j]) for j in range(16)
                    ], dim=1)[None]
                else:
                    filling_latents = masked_latents[0]

                # latent_r[~mask_r] = latent_r[~mask_r] * 0.1 + filling_latents[~mask_r] * 0.9
                latent_r[~mask_r] = filling_latents[~mask_r]

                latents = torch.cat([latent_l, latent_r], 0)
                latents = latents.reshape(b, t, c, h, w)
                img = latents

            # align_steps = [10, 20, 30]
            align_steps = range(10, 49, 5)
            if i in align_steps:
                disparity = torch.nn.functional.interpolate(
                    disparity[0],
                    size=[img.size(-2), img.size(-1)],
                    mode="bicubic",
                    align_corners=True
                )[None]
                strength = 0.1 - 0.1 / len(align_steps) * align_steps.index(i)
                # strength = 0.05
                print("Apply alignment on step", i, "| strength", strength)
                _, align_r = align_stereo_views(
                    img[0].permute(1, 0, 2, 3), img[1].permute(1, 0, 2, 3), disparity[0], stereo_sacle_factor, align_strength=strength)
                align_r = align_r[None].permute(0, 2, 1, 3, 4)
                img = torch.cat([img[:1], align_r], dim=0)
                # img = torch.stack([align_l, align_r], dim=0).permute(0, 2, 1, 3, 4)

            img_prev = img.clone()
            rp_times = 1
            if i in modify_steps and repeat_steps is not None:
                rp_times = [rp_times for (st, rp_times) in repeat_steps if st == i][0]
            print(i, modify_steps, rp_times)
            for j in range(rp_times):
                outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,
                                        mask=mask,x0=x0,fs=fs,guidance_rescale=guidance_rescale,
                                        is_duplicate=is_duplicate,
                                        **kwargs)

                img, pred_x0 = outs
                if rp_times != 1 and j != rp_times - 1:
                    print(f"Apply stereo latent refinement {j}.")
                    img[1:2][mask_r] = img_prev.clone()[1:2][mask_r]
                    img[0:1] = img_prev[0:1].clone()

            if not is_final and (i in modify_steps):
                print("Return to the start")
                intermediates["masked_latents"].append(img[1:2])
                intermediates["masked_latents_left"].append(img[0:1])
                intermediates["mask_r"].append(mask_r)
                return img, intermediates

            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates


def stereo_shift_torch(input_images, depthmaps, sacle_factor=8, shift_both = False, stereo_offset_exponent=1.0):
    '''input: [B, C, H, W] depthmap: [B, H, W]'''

    def _norm_depth(depth,max_val=1):
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = torch.zeros_like(depth)
        return out

    def _create_stereo(input_images, depthmaps, sacle_factor, stereo_offset_exponent):
        c, b, h, w = input_images.shape
        derived_image = torch.zeros_like(input_images)
        sacle_factor_px = (sacle_factor / 100.0) * input_images.shape[-1]

        for batch in range(b):
            for row in range(h):
                # Swipe order should ensure that pixels that are closer overwrite
                # (at their destination) pixels that are less close
                for col in range(w) if sacle_factor_px < 0 else range(w - 1, -1, -1):
                    col_d = col + int((depthmaps[batch, row, col] ** stereo_offset_exponent) * sacle_factor_px)
                    if 0 <= col_d < w:
                        derived_image[:, batch, row, col_d] = input_images[:, batch, row, col]

        return derived_image

    depthmaps = _norm_depth(depthmaps)

    if shift_both is False:
        left = input_images
        balance = 0
    else:
        balance = 0.5
        left = _create_stereo(input_images, depthmaps, + 1 * sacle_factor * balance, stereo_offset_exponent)

    right = _create_stereo(input_images, depthmaps, - 1 * sacle_factor * (1 - balance), stereo_offset_exponent)

    return torch.stack([left, right], axis=0)


def make_latent_grid(latents, fname):
    """latents: (1, 4, 16, 72, 128)"""
    import torchvision
    for i in range(4):
        out = torchvision.utils.make_grid(latents[0, i], 2, normalize=True)[:, None]
        torchvision.utils.save_image(out, f"{fname}_{i}.png")


def align_stereo_views(left_view, right_view, disparity_map, stereo_offset_exponent=1.0, align_strength=0., threshold=0.1):
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

    def _norm_depth(depth,max_val=1):
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = torch.zeros_like(depth)
        return out

    disparity_map = _norm_depth(disparity_map)
    B, C, H, W = left_view.shape
    # Create a meshgrid for warping
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=left_view.device), torch.arange(W, device=left_view.device))
    grid_x = grid_x.unsqueeze(0).unsqueeze(0).float().repeat(B, 1, 1, 1)  # (1, 1, H, W)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0).float().repeat(B, 1, 1, 1)  # (1, 1, H, W)

    # Warp the right view using disparity map
    warped_grid_x = grid_x + stereo_offset_exponent * disparity_map  # Apply disparity to x-coordinate
    warped_grid_x = 2.0 * warped_grid_x / (W - 1) - 1.0  # Normalize to [-1, 1] range for grid_sample
    warped_grid_y = 2.0 * grid_y / (H - 1) - 1.0

    # Combine x and y into a grid for warping
    warped_grid = torch.stack((warped_grid_x, warped_grid_y), dim=-1).squeeze(1)  # (B, H, W, 2)

    # Apply warping to the right view using grid_sample
    left_aligned = F.grid_sample(
        left_view.clone(), warped_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    left_mask = F.grid_sample(
        torch.ones_like(left_view), warped_grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    uncertainty_mask = compute_uncertainty_mask(disparity_map, threshold).bool().repeat(1, 4, 1, 1)
    left_mask[uncertainty_mask] = 0

    aligned_right = (left_aligned * align_strength + right_view * (1 - align_strength)) * left_mask + right_view * (1 - left_mask)
    
    return left_mask, aligned_right