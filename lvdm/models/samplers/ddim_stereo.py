from __future__ import annotations
from typing import Any
import numpy as np
from tqdm import tqdm
import torch
import math
from lvdm.models.utils_diffusion import make_ddim_sampling_parameters, make_ddim_timesteps, rescale_noise_cfg
from lvdm.common import noise_like
from lvdm.common import extract_into_tensor
from lvdm.models.samplers.ddim import DDIMSampler
from einops import rearrange
import copy
import kornia
import torchvision.transforms as transforms
import torch.nn.functional as F

from lvdm.models.samplers.stereo_utils import (
    stereo_shift_torch, align_stereo_views, warp_fft_low_frequency, create_stereo,  _norm_depth,
)
from lvdm.models.samplers.depth_utils import compute_uncertainty_mask
from lvdm.models.samplers.inpainting_utils import Inpainting

import torch


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
                      stereo_scale_factor=8, masked_latents=None, masked_latents_weight=0.7, is_final=True, cur_iter=None, total_iter=None, modify_step=None,
                      fuse_latent=False, stereo_repeat_times=1, repeat_steps=None, mimic_final = False,
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
                        cond_copy.update({key: cond[key].repeat(2, *([1] * (len(cond[key].shape) - 1)))})
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

        intermediates = {'x_inter': [img], 'pred_x0': [img], "masked_latents": [], "masked_latents_left": [], "init_xT": [img], "mask_r": [], "others":[]}

        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]

        if verbose:
            iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        else:
            iterator = time_range
 
        clean_cond = kwargs.pop("clean_cond", False)

        left_latents = kwargs.get("left_latents", None)

        disparity_estimated = kwargs.get("disparity_estimated", None)
        disparity_prop = disparity

        # cond_copy, unconditional_conditioning_copy = copy.deepcopy(cond), copy.deepcopy(unconditional_conditioning)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if step_and_latent is not None:
                if i < step_and_latent[0] + 1:
                    print(f"skip step {i} / {step_and_latent[0]}")
                    noise_like(img.shape, device, repeat=False)
                    continue
                if i == step_and_latent[0] + 1:
                    img = step_and_latent[1]
                    mask_r = step_and_latent[2]

            # disparity = disparity_prop if i < 20 else disparity_estimated

            if (kwargs.get("warped_cond", None) and i < 6):
                imtext_cond_warped = kwargs.get("imtext_cond_warped", None)
                img_tensor_repeat_warped = kwargs.get("img_tensor_repeat_warped", None)

                if is_duplicate and img_tensor_repeat_warped is not None:
                    print("c_concat")
                    ctensor = cond["c_concat"][0]
                    cond["c_concat"] = [torch.cat([ctensor[0:1], img_tensor_repeat_warped[0:1]], dim=0)]

                if is_duplicate and imtext_cond_warped is not None:
                    if unconditional_guidance_scale == 1:
                        print("c_crossattn")
                        ctensor = cond["c_crossattn"][0]
                        cond["c_crossattn"] = [torch.cat([ctensor[0:1], imtext_cond_warped[0:1]], dim=0)]

            # elif kwargs.get("warped_cond", None):
            else:

                ctensor = cond["c_concat"][0]
                cond["c_concat"] = [torch.cat([ctensor[0:1], ctensor[0:1]], dim=0)]

                if unconditional_guidance_scale == 1:
                    ctensor = cond["c_crossattn"][0]
                    cond["c_crossattn"] = [torch.cat([ctensor[0:1], ctensor[0:1]], dim=0)]

            ## use mask to blend noised original latent (img_orig) & new sampled latent (img)
            if mask is not None:
                assert x0 is not None
                if clean_cond:
                    img_orig = x0
                else:
                    img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass? <ddim inversion>
                img = img_orig * mask + (1. - mask) * img # keep original & modify use img

            if isinstance(modify_step, (list, tuple)):
                modify_steps = modify_step
            else:
                modify_steps = [] if modify_step is None else [modify_step]

            # if repeat_steps is not None:
            #     modify_steps = [st for (st, rp_times) in repeat_steps]

            if i in modify_steps:
                print(f"Apply modification on step {i}")
                b, c, t, h, w = img.shape
                assert b == 2
                scale_factor = stereo_scale_factor
                shift_both = False
                # NOTE: remove more details
                # disparity = torch.nn.functional.interpolate(
                #     disparity[0],
                #     size=[img.size(-2) // 4, img.size(-1) // 4],
                #     mode="nearest"
                # )[None]
                disparity = torch.nn.functional.interpolate(
                    disparity[0],
                    size=[img.size(-2), img.size(-1)],
                    mode="bilinear",
                    align_corners=False,
                )[None]
                latents = img
                fft = kwargs.get("fft", None)
                if fft:
                    print("Warp with fft.")
                    latents_ts = warp_fft_low_frequency(latents, disparity, scale_factor)
                else:
                    latents_ts = stereo_shift_torch(
                        latents[0], disparity[0, :, 0], scale_factor=scale_factor, shift_both=shift_both)

                if shift_both:
                    latent_l = latents_ts[0:1]  # Shape: (torch.Size([1, 4, 16, 72, 128]))
                    latent_r = latents_ts[1:2]  # Shape: (torch.Size([1, 4, 16, 72, 128]))
                else:
                    latent_l = latents[0:1]
                    latent_r = latents_ts[1:2]

                # make_latent_grid(latent_l.abs(), "latent_l")
                # make_latent_grid(latent_r.abs(), "latent_r")
                print(latent_r.shape)
                mask_r = latent_r.reshape(-1, c, t, h, w)[:, 0, :, ...] != 0
                mask_r = rearrange(mask_r, 'b t h w -> b () t h w').repeat(1, 4, 1, 1, 1)
                mask_l = latent_l.reshape(-1, c, t, h, w)[:, 0, :, ...] != 0
                mask_l = rearrange(mask_l, 'b t h w -> b () t h w').repeat(1, 4, 1, 1, 1)

                filling_latents: torch.Tensor
                if masked_latents is None:
                    if i == 0:
                        filling_latents = latent_l
                        print("Fill with left")
                    else:
                        # filling_latents = latents[1:2]
                        filling_latents = latent_r
                        print("Fill with right")
                else:
                    try:
                        filling_latents = masked_latents[modify_steps.index(i)]
                    except:
                        print(modify_steps, i, "use the last one instead")
                        filling_latents = masked_latents[-1]

                # Init the first interation only
                clean_border = kwargs.get("clean_border", False)
                if clean_border:
                    blank_column_indices = (mask_r[:, 0].sum(dim=1) < 16).all(dim=-2)
                    if stereo_scale_factor > 0:
                        blank_start = blank_column_indices.flip(-1).float().argmin(dim=-1)
                    else:
                        blank_start = blank_column_indices.float().argmin(dim=-1)
                    print("border", blank_start)
                    # Create a mask for inpainting
                    mask_border = torch.zeros_like(mask_r).to(filling_latents)
                    for ii, start in enumerate(blank_start):
                        if start > 0:  # If there are blank columns
                            if stereo_scale_factor > 0:
                                mask_border[ii, ..., - start:] = 1  # Mask only the detected blank columns
                            else:
                                mask_border[ii, ..., :start] = 1  # Mask only the detected blank columns
                    mask_border_valid = mask_border
                    print(mask_border.shape, filling_latents.shape, mask_border_valid.shape)
                    # make_latent_grid(mask_border_valid, "mask_border_valid")
                    lam = 0.9
                    filling_latents = filling_latents * (1 - mask_border_valid) + mask_border_valid * (latent_l * lam + filling_latents * (1 - lam))

                if is_final or mimic_final:
                    try:
                        filling_latents = masked_latents[modify_steps.index(i)]
                    except:
                        print("Overwrite filling latents with the right view.", img.shape)
                        filling_latents = img[1:2]

                # make_latent_grid(latent_r.abs(), f"latent_r_masked_{i}")
                latent_r[~mask_r] = filling_latents[~mask_r]

                latent_r_copy = latent_r.clone()
                latent_r_copy[mask_r] = 0
                noisy_restart_2 = kwargs.get("noisy_restart_2", None)
                if noisy_restart_2:
                    print("noisy_restart_2")
                    latent_r = kornia.filters.joint_bilateral_blur(
                        latent_r[0].clone(), latent_l[0].clone(), (3, 3), 0.1, (1.5, 1.5))[None].to(latent_r)
                    # make_latent_grid(latent_r, "latent_r_masked_blurred")

                latents = torch.cat([latent_l, latent_r], 0)
                latents = latents.reshape(b, c, t, h, w)
                img = latents

            max_warp_step = kwargs.get("max_warp_step", 45)

            rp_times = 1
            rp_steps = []
            if repeat_steps is not None:
                rp_steps = [st for (st, rp_times) in repeat_steps]
            if i in rp_steps:
                rp_times = [rp_times for (st, rp_times) in repeat_steps if st == i][0]
            sigmas = self.ddim_sigmas_for_original_num_steps if ddim_use_original_steps else self.ddim_sigmas

            all_predictions = []
            for j in range(rp_times):
                if j == 0:
                    img_prev = img.clone()
                    if step_and_latent and i == step_and_latent[0] + 1:
                        _, pred_x0, a_prev_sqrt, dir_xt, noise_out = kwargs.get("others")
                    if rp_times > 1:
                        pred_x0_prev = pred_x0.clone()
                        a_prev_sqrt_prev = a_prev_sqrt.clone()
                        dir_xt_prev = dir_xt.clone()
                        noise_out_prev = noise_out.clone()
                        mean_before = pred_x0_prev.mean(dim=[0, 2, 3, 4], keepdim=True)
                        std_before = pred_x0_prev.std(dim=[0, 2, 3, 4], keepdim=True)
                        img_prev_updated_r = img_prev.clone()[1:2]
                if j != 0:
                    propagation = kwargs.get("propagation")
                    print(f"Apply stereo latent refinement {j}.")
                    option = "direct_on_img"
                    # Rescale_noise
                    sigma_t = torch.full((b, 1, 1, 1, 1), sigmas[index], device=device)
                    sigma_t_1 = torch.full((b, 1, 1, 1, 1), sigmas[index + 1], device=device)
                    scale_factor = sigma_t / sigma_t_1
                    if option == "direct_on_img":
                        img = img * scale_factor[1:2]
                        if i > max_warp_step:
                            print("Refine other parts on", i)

                            img_prev_updated_r[~mask_r] = img[1:2][~mask_r]
                        elif i >= max_warp_step:
                            img_prev_updated_r[~mask_r] = img[1:2][~mask_r]

                        elif i >= 10:
                            img_prev_updated_r[~mask_r] = img[1:2][~mask_r]
                        else:

                            img_prev_updated_r[~mask_r] = img[1:2][~mask_r]

                        img = torch.cat([img_prev.clone()[0:1], img_prev_updated_r])

                    elif option == "on_pred_x0_only":
                        alphas_prev = self.model.alphas_cumprod_prev if ddim_use_original_steps else self.ddim_alphas_prev
                        n_weight = torch.as_tensor(alphas_prev[index]).sqrt().to(img)
                        _mask_r = mask_r.repeat(2, 1, 1, 1, 1)
                        pred_x0_updated_r = pred_x0_prev.clone()
                        print(i, n_weight)
                        pred_x0_updated_r[~_mask_r] = (pred_x0[~_mask_r] * n_weight + pred_x0_updated_r[~_mask_r] * (1 - n_weight))

                        img_updated = a_prev_sqrt_prev * pred_x0_updated_r + dir_xt_prev + noise_out_prev
                        img = torch.cat([img_prev.clone()[0:1], img_updated[1:2]])
                    else:
                        raise RuntimeError

                aa = cond["c_crossattn"][0][:1]
                cond.update({"c_crossattn": [torch.cat([aa, aa])]})

                outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,
                                        mask=mask,x0=x0,fs=fs,guidance_rescale=guidance_rescale,
                                        is_duplicate=is_duplicate, stereo_scale_factor=stereo_scale_factor,
                                        output_all=True,
                                        **kwargs)
                img, pred_x0, a_prev_sqrt, dir_xt, noise_out = outs

                all_predictions.append(img.clone())

            if left_latents is not None:
                img[0:1] = left_latents[i + 1][0:1].clone()

            output_step = kwargs.get("output_step", None)
            if not is_final and (i in [output_step]):
                intermediates["masked_latents_left"].append(img[0:1])
                intermediates["masked_latents"].append(img[1:2])
                intermediates["mask_r"].append(mask_r)
                intermediates["others"].append((img, pred_x0, a_prev_sqrt, dir_xt, noise_out))
            elif not is_final and (i in modify_steps):
                intermediates["masked_latents_left"].append(img[0:1])
                noisy_restart = kwargs.get("noisy_restart", None)
                if noisy_restart is not None and noisy_restart:
                    alphas = self.model.alphas_cumprod if ddim_use_original_steps else self.ddim_alphas
                    alphas_prev = self.model.alphas_cumprod_prev if ddim_use_original_steps else self.ddim_alphas_prev
                    sigmas = self.ddim_sigmas_for_original_num_steps if ddim_use_original_steps else self.ddim_sigmas
                    n_weight = torch.as_tensor(alphas[index]).sqrt().to(img)
                    # n_weight = torch.as_tensor(alphas[total_steps - index]).to(img)
                    sigma_t = torch.full((b, 1, 1, 1, 1), sigmas[index], device=device)
                    if index + 1 == 50:
                        scale_factor = torch.ones_like(sigma_t)
                    else:
                        sigma_t_1 = torch.full((b, 1, 1, 1, 1), sigmas[index + 1], device=device)
                        scale_factor = sigma_t / sigma_t_1
                    fixed_noise_2 = kwargs.get("fixed_noise_2")
                    if fixed_noise_2 is None:
                        noise = sigma_t * noise_like(img.shape, device, False)
                    else:
                        noise = sigma_t * fixed_noise_2
                    intermediates["masked_latents"].append(img[1:2] * (1 - n_weight) * scale_factor[1:2] + noise[1:2] * (n_weight))
                    print("A", "sigma", alphas_prev[index], alphas[index], n_weight)
                else:
                    intermediates["masked_latents"].append(img[1:2])
                    print("B")
                intermediates["mask_r"].append(mask_r)
                intermediates["others"].append((img, pred_x0, a_prev_sqrt, dir_xt, noise_out))

                reuse_modify_latents = kwargs.get("reuse_modify_latents", None)
                if reuse_modify_latents:
                    if modify_steps.index(i) == 0 and masked_latents is not None: 
                        print("`masked_latents` provided. Skip reuse.")
                    else:
                        if masked_latents is None:
                            masked_latents = [img[1:2], img[1:2]]  # Make two steps for the first iter.
                        else:
                            masked_latents.append(img[1:2])

            if not is_final and output_step is not None and (i in [output_step]):
                if len(intermediates["masked_latents"]) == len(modify_steps):
                    print("Return to the start", len(intermediates["masked_latents"]))
                    return img, intermediates
            elif not is_final and output_step is None and (i in modify_steps):
                if len(intermediates["masked_latents"]) == len(modify_steps):
                    print("Return to the start", len(intermediates["masked_latents"]))
                    return img, intermediates

            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            intermediates['x_inter'].append(img)
            intermediates['pred_x0'].append(pred_x0)

        return img, intermediates


def make_latent_grid(latents, fname):
    """latents: (1, 4, 16, 72, 128)"""
    import torchvision
    for i in range(4):
        out = torchvision.utils.make_grid(latents[0, i], 2, normalize=True)[:, None]
        torchvision.utils.save_image(out, f"{fname}_{i}.png")
