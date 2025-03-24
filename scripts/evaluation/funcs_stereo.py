import os, sys
import numpy as np

from PIL import Image
import io

import torch
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.samplers.ddim_stereo import DDIMStereoSampler
from transformers import pipeline
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
import torch.nn.functional as F
import kornia
import torchvision
from pytorch_lightning import seed_everything
from lvdm.common import noise_like

from scripts.gradio.stereoutils import BNAttention, regiter_attention_editor_diffusers


def save_depth_video(vid_tensor, savepath, fps):
    print(vid_tensor.shape, vid_tensor.min(), vid_tensor.max(), savepath)
    video = vid_tensor.detach().cpu() # stack in temporal dim [t, 1, h, w]
    if video.shape[1] == 1:
        video = kornia.color.grayscale_to_rgb(video)
    grid = (video * 255).to(torch.uint8).permute(0, 2, 3, 1)
    torchvision.io.write_video(savepath, grid, fps=fps, video_codec='h264', options={'crf': '10'})


def compute_initial_depth(images):
    depth_array = []
    depth_pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-large-hf")
    for i, image in enumerate(images):
        image = ToPILImage()(image)
        depth_image = depth_pipe(image)["depth"]
        depth = torch.as_tensor(np.array(depth_image), device="cuda")[None]
        depth_array.append(depth)
    return torch.stack(depth_array, dim=0)


def compute_depthcrafter(images, save_name=None, num_denoising_steps_dissovling=50, dissolve_step=None):
    from depthcrafter import DepthCrafterWrapper
    images = images.permute(0, 2, 3, 1)
    model = DepthCrafterWrapper(
        unet_path="tencent/DepthCrafter",
        pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
        cpu_offload="model",
    )
    if dissolve_step is not None:
        return torch.tensor(
            model.infer_frames(
                images.cpu().numpy(),
                save_to=save_name,
                num_denoising_steps=5,
                num_denoising_steps_dissovling=num_denoising_steps_dissovling,
                dissolve_step=dissolve_step
            ), device=images.device, dtype=images.dtype
        )[:, None]

    return torch.tensor(
        model.infer_frames(images.cpu().numpy(), save_to=save_name, num_denoising_steps=5), device=images.device, dtype=images.dtype
    )[:, None]


def dissolve_depth(images, save_name=None, num_denoising_steps_dissovling=50, dissolve_step=None):
    from depthcrafter import DepthCrafterWrapper
    images = images.permute(0, 2, 3, 1)
    model = DepthCrafterWrapper(
        unet_path="tencent/DepthCrafter",
        pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
        cpu_offload="model",
    )
    if dissolve_step is not None:
        return torch.tensor(
            model.infer_frames(
                images.cpu().numpy(),
                save_to=save_name,
                num_denoising_steps=5,
                num_denoising_steps_dissovling=num_denoising_steps_dissovling,
                dissolve_step=dissolve_step
            ), device=images.device, dtype=images.dtype
        )[:, None]

    return torch.tensor(
        model.infer_frames(images.cpu().numpy(), save_to=save_name, num_denoising_steps=5), device=images.device, dtype=images.dtype
    )[:, None]


def compute_video_depth_anything(images, save_name=None):
    from video_depthanything import VideoDepthAnythingWrapper
    images = images.permute(0, 2, 3, 1)
    model = VideoDepthAnythingWrapper()
    return torch.tensor(
        model.infer_frames((images.cpu().numpy() * 255.).astype(np.uint8), save_to=save_name), device=images.device, dtype=images.dtype
    )[:, None]


from PIL import Image
import io


def load_PIL(
    img_pil: Image.Image, auto_rotate: bool = True, remove_alpha: bool = True
):
    """Load an RGB image.

    Args:
    ----
        path: The url to the image to load.
        auto_rotate: Rotate the image based on the EXIF data, default is True.
        remove_alpha: Remove the alpha channel, default is True.

    Returns:
    -------
        img: The image loaded as a numpy array.
        icc_profile: The color profile of the image.
        f_px: The optional focal length in pixels, extracting from the exif data.

    """
    import depth_pro
    import depth_pro.utils

    img_exif = depth_pro.utils.extract_exif(img_pil)
    icc_profile = img_pil.info.get("icc_profile", None)

    # Rotate the image.
    if auto_rotate:
        exif_orientation = img_exif.get("Orientation", 1)
        if exif_orientation == 3:
            img_pil = img_pil.transpose(Image.ROTATE_180)
        elif exif_orientation == 6:
            img_pil = img_pil.transpose(Image.ROTATE_270)
        elif exif_orientation == 8:
            img_pil = img_pil.transpose(Image.ROTATE_90)
        elif exif_orientation != 1:
            print(f"Ignoring image orientation {exif_orientation}.")

    img = np.array(img_pil)
    # Convert to RGB if single channel.
    if img.ndim < 3 or img.shape[2] == 1:
        img = np.dstack((img, img, img))

    if remove_alpha:
        img = img[:, :, :3]

    # Extract the focal length from exif data.
    f_35mm = img_exif.get(
        "FocalLengthIn35mmFilm",
        img_exif.get(
            "FocalLenIn35mmFilm", img_exif.get("FocalLengthIn35mmFormat", None)
        ),
    )
    if f_35mm is not None and f_35mm > 0:
        f_px = depth_pro.utils.fpx_from_f35(img.shape[1], img.shape[0], f_35mm)
    else:
        f_px = None

    return img, icc_profile, f_px


def compute_initial_depth_pro(images):
    import depth_pro

    depth_array = []
    focallength_px_array = []

    model, transform = depth_pro.create_model_and_transforms()
    model.cuda().eval()

    for i, image in enumerate(images):

        np_img = image.cpu().permute(1, 2, 0).numpy()
        np_img = (np_img * 255).astype(np.uint8)

        # Create a PIL image from the numpy array
        pil_img = Image.fromarray(np_img)

        # Load and preprocess an image.
        image, _, f_px = load_PIL(pil_img)
        image = transform(image)
        # Run inference.
        prediction = model.infer(image.cuda(), f_px=f_px.cuda() if f_px is not None else None)
        depth = prediction["depth"]  # Depth in [m].
        focallength_px = prediction["focallength_px"]  # Focal length in pixels.

        depth_array.append(focallength_px / depth[None])
        focallength_px_array.append(focallength_px)

    del model
    del transform
    depths = torch.stack(depth_array, dim=0)
    return (depths - depths.min()) / (depths.max() - depths.min())


def raft_disp(images, resize=None):
    if resize is not None:
        images = F.interpolate(images, resize)
    # images = kornia.grad_estimator.ste.STEFunction.apply(latents, images)
    # images = torch.flip(images, dims=(0,))
    raft = torchvision.models.optical_flow.raft_large(
        torchvision.models.optical_flow.Raft_Large_Weights.DEFAULT
    ).cuda().eval()
    return raft(images[:-1], images[1:])[-1]


def gmflow_disp(images, resize=None, direction="forward", flow_model='sea_raft_m', pretrained_ckpt='things'):
    # if resize is not None:
    #     images = F.interpolate(images, scale_factor=0.5)
    images = F.interpolate(images, scale_factor=0.5)
    import ptlflow
    # from ptlflow.utils import flow_utils
    # from ptlflow.utils.io_adapter import IOAdapter
    # model = ptlflow.get_model('neuflow2', pretrained_ckpt='mixed').cuda().eval()
    # model = ptlflow.get_model('memflow', pretrained_ckpt='spring').cuda().eval()
    model = ptlflow.get_model(flow_model, pretrained_ckpt=pretrained_ckpt).cuda().eval()
    # model = ptlflow.get_model('memflow_t', pretrained_ckpt='things').cuda().eval()
    # flow_model = 'videoflow_mof'
    # model = ptlflow.get_model(flow_model, pretrained_ckpt='sintel').cuda().eval()
    images = kornia.color.rgb_to_bgr(images)
    if direction == "forward":
        if flow_model.startswith('videoflow'):
            x = torch.stack([images[:-1], images[1:], torch.cat([images[2:], images[-1:]], dim=0)], dim=1)
            out = model({"images": x})["flows"]
        else:
            x = torch.stack([images[:-1], images[1:]], dim=1)
    else:
        if flow_model.startswith('videoflow'):
            x = torch.stack([torch.cat([images[2:], images[-1:], images[1:], images[:-1]], dim=0)], dim=1)
            out = model({"images": x})["flows"]
        else:
            x = torch.stack([images[1:], images[:-1]], dim=1)
        
    out = model({"images": x})["flows"][:, 0]
    out = F.interpolate(out, scale_factor=2)
    return out


def smooth_optical_flow_kornia(flow, kernel_size=(5, 5), sigma=(1.0, 1.0)):
    """Smooths an optical flow tensor using Kornia's Gaussian filter."""
    # Kornia expects the kernel size and sigma as tuples for 2D Gaussian
    flow_u = flow[:, 0:1, :, :]  # Horizontal component (u)
    flow_v = flow[:, 1:2, :, :]  # Vertical component (v)
    
    # Apply Kornia's Gaussian blur to each flow component
    flow_u_smooth = kornia.filters.gaussian_blur2d(flow_u, kernel_size, sigma)
    flow_v_smooth = kornia.filters.gaussian_blur2d(flow_v, kernel_size, sigma)
    
    # Combine the smoothed components
    smoothed_flow = torch.cat([flow_u_smooth, flow_v_smooth], dim=1)
    return smoothed_flow


def median_blur_optical_flow(flow, kernel_size=5):
    """Applies Kornia's median blur to smooth an optical flow tensor."""
    # Apply median blur to each component of the flow
    flow_u = flow[:, 0:1, :, :]  # Horizontal component (u)
    flow_v = flow[:, 1:2, :, :]  # Vertical component (v)
    
    # Apply Kornia's median blur
    flow_u_blurred = kornia.filters.median_blur(flow_u, (kernel_size, kernel_size))
    flow_v_blurred = kornia.filters.median_blur(flow_v, (kernel_size, kernel_size))
    
    # Combine the blurred components
    blurred_flow = torch.cat([flow_u_blurred, flow_v_blurred], dim=1)
    return blurred_flow


def detect_occlusions(flow_forward, flow_backward, threshold=1.0):
    """
    Detect occlusions using forward-backward flow consistency.
    
    Args:
        flow_forward (torch.Tensor): Optical flow from frame t to t+1, shape [B, 2, H, W]
        flow_backward (torch.Tensor): Optical flow from frame t+1 to t, shape [B, 2, H, W]
        threshold (float): Threshold for occlusion detection
    
    Returns:
        occlusion_mask (torch.Tensor): Occlusion mask, shape [B, 1, H, W]
    """
    B, _, H, W = flow_forward.shape
    device = flow_forward.device
    
    # Create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    grid_x = grid_x.unsqueeze(0).expand(B, H, W).float()
    grid_y = grid_y.unsqueeze(0).expand(B, H, W).float()
    
    # Warp grid using forward flow
    flow_f_x = flow_forward[:, 0, :, :]
    flow_f_y = flow_forward[:, 1, :, :]
    warped_x = grid_x + flow_f_x
    warped_y = grid_y + flow_f_y
    
    # Sample backward flow at warped positions
    flow_b_x = F.grid_sample(
        flow_backward[:, 0, :, :].unsqueeze(1),
        torch.stack((2 * warped_x / (W - 1) - 1, 2 * warped_y / (H - 1) - 1), dim=-1),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    ).squeeze(1)
    flow_b_y = F.grid_sample(
        flow_backward[:, 1, :, :].unsqueeze(1),
        torch.stack((2 * warped_x / (W - 1) - 1, 2 * warped_y / (H - 1) - 1), dim=-1),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    ).squeeze(1)
    
    # Compute flow consistency error
    flow_diff_x = flow_b_x + flow_f_x
    flow_diff_y = flow_b_y + flow_f_y
    flow_diff = torch.sqrt(flow_diff_x ** 2 + flow_diff_y ** 2)
    
    # Occlusion mask
    occlusion_mask = (flow_diff > threshold).unsqueeze(1).float()  # Shape: [B, 1, H, W]
    
    return occlusion_mask


def propagate_disparity_warping_precompute(initial_disparity, disparity, disparity_back, init_index=0):
    """Propagate disparity maps across frames using feature matching."""
    T, C, H, W = disparity.shape
    
    occlusion_mask = detect_occlusions(disparity, disparity_back)
    occlusion_mask = F.interpolate(
        occlusion_mask, size=initial_disparity.shape[-2:], mode="nearest")
    occlusion_mask = occlusion_mask.bool()

    def _propagate(init_index, direction):
        propagated_disparities = [initial_disparity]
        warped_disparities = [initial_disparity]
        for t in range(init_index, T) if direction == "forward" else range(init_index - 1, -1, -1):

            # Estimate disparity between consecutive frames
            estimated_movement = disparity[t, :][None]
            estimated_movement = F.interpolate(
                estimated_movement, size=initial_disparity.shape[-2:], mode="bilinear", align_corners=False)

            output_disparity = warp_depth_map_v2(propagated_disparities[-1], estimated_movement, direction)
            propagated_disparities.append(output_disparity)

            # warped_disparity = warp_depth_map_v2(warped_disparities[-1], estimated_movement, direction)
            output_disparity = restore_occluded_pixels(output_disparity, propagated_disparities[-2:])
            warped_disparities.append(output_disparity)

        return warped_disparities
    
    backward_propagated_disparities_tensor = _propagate(init_index, direction="backward")[1:][::-1]
    forward_propagated_disparities_tensor = _propagate(init_index, direction="forward")

    # Stack the list to form a tensor of shape BxTx1xHxW
    return torch.stack([*backward_propagated_disparities_tensor, *forward_propagated_disparities_tensor], dim=1)


def restore_occluded_pixels(warped_depth_map, previous_depth_map, occluded_mask=None):
    """
    Restore pixels that were set to zero after warping by using the previous depth map.
    
    Args:
        warped_depth_map (torch.Tensor): The warped depth map, shape [B, 1, H, W].
        previous_depth_map (torch.Tensor): The depth map from the previous frame, shape [B, 1, H, W].
    
    Returns:
        torch.Tensor: The depth map with occluded pixels restored.
    """
    if occluded_mask is None:
        occluded_mask = (warped_depth_map == 0)
    
    restored_depth_map = warped_depth_map.clone()
    for depth_map in previous_depth_map:
        restored_depth_map[occluded_mask] = depth_map[occluded_mask]
        occluded_mask = (restored_depth_map == 0) & occluded_mask
    return restored_depth_map


def warp_depth_map_v2(depth_map, flow, direction):
    B, C, H, W = depth_map.size()
    device = depth_map.device

    # Create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    grid_x = grid_x.unsqueeze(0).expand(B, H, W)  # Shape: [B, H, W]
    grid_y = grid_y.unsqueeze(0).expand(B, H, W)  # Shape: [B, H, W]

    # Extract flow components
    flow_x = flow[:, 0, :, :]
    flow_y = flow[:, 1, :, :]

    # Adjust grid coordinates based on flow and direction
    if direction == "forward":
        dest_x = grid_x + flow_x
        dest_y = grid_y + flow_y
    else:
        dest_x = grid_x - flow_x
        dest_y = grid_y - flow_y

    # Round destination coordinates to nearest integer
    dest_x = dest_x.round().long()
    dest_y = dest_y.round().long()

    # Create mask for valid destination coordinates
    valid_mask = (dest_x >= 0) & (dest_x < W) & (dest_y >= 0) & (dest_y < H)

    # Filter valid coordinates
    src_batch_inds = torch.arange(B, device=device).view(B, 1, 1).expand(B, H, W)[valid_mask]
    src_y = grid_y[valid_mask]
    src_x = grid_x[valid_mask]
    dest_y = dest_y[valid_mask]
    dest_x = dest_x[valid_mask]

    # Compute flattened indices for destination
    dest_flat_inds = src_batch_inds * (H * W) + dest_y * W + dest_x

    # Reverse indices to prioritize the last assignments
    rev_src_batch_inds = src_batch_inds.flip(0)
    rev_src_y = src_y.flip(0)
    rev_src_x = src_x.flip(0)
    rev_dest_flat_inds = dest_flat_inds.flip(0)

    # Replace torch.unique with NumPy's unique function
    rev_dest_flat_inds_np = rev_dest_flat_inds.cpu().numpy()
    _, unique_indices_np = np.unique(rev_dest_flat_inds_np, return_index=True)
    unique_indices = torch.from_numpy(unique_indices_np).to(device)

    # Extract the final indices for assignments
    final_batch_inds = rev_src_batch_inds[unique_indices]
    final_src_y = rev_src_y[unique_indices]
    final_src_x = rev_src_x[unique_indices]
    final_dest_y = dest_y.flip(0)[unique_indices]
    final_dest_x = dest_x.flip(0)[unique_indices]

    # Prepare indices for assignment
    final_batch_inds = final_batch_inds.long()
    final_src_y = final_src_y.long()
    final_src_x = final_src_x.long()
    final_dest_y = final_dest_y.long()
    final_dest_x = final_dest_x.long()

    # Initialize warped depth map and mask
    warped_depth = torch.zeros_like(depth_map)
    # warped_mask = torch.zeros_like(depth_map)

    # Assign depth values using advanced indexing
    warped_depth[final_batch_inds, :, final_dest_y, final_dest_x] = \
        depth_map[final_batch_inds, :, final_src_y, final_src_x]
    # warped_mask[final_batch_inds, :, final_dest_y, final_dest_x] = 1.0

    return warped_depth


def propagate_all_frames(depth, disparity_opt, disparity_opt_back, frame_interval=1, kernel_size=5, N=3):
    # depth = compute_initial_depth_pro(batch_images[7:8]) * 255
    # disparity_prop = propagate_disparity_warping_precompute(depth / 255, disparity_opt, disparity_opt_back, init_index=7)
    # depth = compute_initial_depth_pro(torch.cat([batch_images[2:3], batch_images[7:8], batch_images[12:13]])) * 255
    disparity_all = []

    for i in range(len(depth) // frame_interval):
        i *= frame_interval
        disparity_prop_i = propagate_disparity_warping_precompute(
            depth[i:i + 1] / 255, disparity_opt, disparity_opt_back, init_index=i
        )
        disparity_all.append(disparity_prop_i)

    out = torch.stack(disparity_all)

    return out.mean(dim=0)

    # Create a new tensor to store the summed frames
    result = torch.zeros(out.size(0) - N + 1, *out.size()[1:], device=out.device, dtype=out.dtype)
    # Sum N adjacent frames
    for i in range(result.size(0)):
        result[i] = out[i:i + N].sum(dim=0)

    # if kernel_size is not None:
    #     out[0] = kornia.filters.median_blur(out[0], (kernel_size, kernel_size))
    return result.mean(dim=0)


@torch.no_grad()
def batch_stereo_ddim_sampling(
    model, cond, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1.0,
    cfg_scale=1.0, temporal_cfg_scale=None, stereo_att_start_step=25, stereo_att_end_step=45, stereo_scale_factor=8,
    prompt_str=None, imtext_cond_warped=None, img_tensor_repeat_warped=None, max_warp_step=40, scale_disparity_factor=1,
    refine_non_mask=False, seed=None, propagation=False, dissolve_step=None,
    **kwargs
):
    ddim_sampler = DDIMSampler(model)
    ddim_stereo_sampler = DDIMStereoSampler(model)
    uncond_type = model.uncond_type
    batch_size = noise_shape[0]
    fs = cond["fs"]
    del cond["fs"]
    if noise_shape[-1] == 32:
        timestep_spacing = "uniform"
        guidance_rescale = 0.0
    else:
        timestep_spacing = "uniform_trailing"
        guidance_rescale = 0.7
    ## construct unconditional guidance
    if cfg_scale != 1.0:
        if uncond_type == "empty_seq":
            prompts = batch_size * [""]
            #prompts = N * T * [""]  ## if is_imgbatch=True
            uc_emb = model.get_learned_conditioning(prompts)
        elif uncond_type == "zero_embed":
            c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
            uc_emb = torch.zeros_like(c_emb)

        ## process image embedding token
        if hasattr(model, 'embedder'):
            uc_img = torch.zeros(noise_shape[0],3,224,224).to(model.device)
            ## img: b c h w >> b l c
            uc_img = model.embedder(uc_img)
            uc_img = model.image_proj_model(uc_img)
            uc_emb = torch.cat([uc_emb, uc_img], dim=1)

        if isinstance(cond, dict):
            uc = {key:cond[key] for key in cond.keys()}
            uc.update({'c_crossattn': [uc_emb]})
        else:
            uc = uc_emb
    else:
        uc = None

    batch_variants = []

    for _ in range(n_samples):
        assert n_samples == 1
        if ddim_sampler is not None:
            kwargs.update({"clean_cond": True})
            samples, intermediates = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=noise_shape[0],
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=cfg_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            temporal_length=noise_shape[2],
                                            conditional_guidance_scale_temporal=temporal_cfg_scale,
                                            fs=fs,
                                            timestep_spacing=timestep_spacing,
                                            guidance_rescale=guidance_rescale,
                                            is_duplicate=False,
                                            log_every_t=1,
                                            **kwargs
                                            )
        left_latents = intermediates['x_inter']
        ## reconstruct from latent to pixel space
        batch_images: torch.Tensor = model.decode_first_stage(samples)

        B, C, T, H, W = batch_images.shape
        batch_images = (batch_images.permute(0, 2, 1, 3, 4).view(-1, C, H, W) + 1) / 2
        batch_images = (torch.clamp(batch_images, -1, 1) + 1) / 2
        print(batch_images.shape, batch_images.min(), batch_images.max())
        # # Disparity propagation
        # depth = compute_initial_depth(ToPILImage()(batch_images[0]))
        # disparity_opt = raft_disp(batch_images)
        # disparity_prop = propagate_disparity_warping_precompute(depth[None], disparity_opt)

        # Sequential
        # disparity_prop = compute_initial_depth(batch_images)[None]
        # disparity_prop = compute_initial_depth_pro(batch_images)[None]

        if propagation in ["depth_pro", "depth_anything"]:
            # depth = compute_initial_depth(batch_images[0])
            # disparity_opt = raft_disp(batch_images, resize=(256, 384))
            disparity_opt = gmflow_disp(batch_images)
            disparity_opt_back = gmflow_disp(batch_images, direction="backward")

        if propagation == "depth_pro":
            # depth = compute_initial_depth_pro(batch_images[7:8]) * 255
            # disparity_prop = propagate_disparity_warping_precompute(depth / 255, disparity_opt, disparity_opt_back, init_index=7)
            # depth = compute_initial_depth_pro(torch.cat([batch_images[2:3], batch_images[7:8], batch_images[12:13]])) * 255
            # disparity_prop_a = propagate_disparity_warping_precompute(depth[0:1] / 255, disparity_opt[:5], disparity_opt_back[:5], init_index=2)
            # disparity_prop_b = propagate_disparity_warping_precompute(depth[1:2] / 255, disparity_opt[5:10], disparity_opt_back[5:10], init_index=2)
            # disparity_prop_c = propagate_disparity_warping_precompute(depth[2:3] / 255, disparity_opt[10:], disparity_opt_back[10:], init_index=2)
            # disparity_prop_b[:, 0:1] = (disparity_prop_b[:, 0:1] + disparity_prop_a[:, 5:6]) / 2
            # disparity_prop_c[:, 0:1] = (disparity_prop_c[:, 0:1] + disparity_prop_b[:, 5:6]) / 2
            # disparity_prop = torch.cat([disparity_prop_a[:, :5], disparity_prop_b[:, :5], disparity_prop_c], dim=1)
            depth = compute_initial_depth_pro(batch_images) * 255
            # depth = 1 / (depth + 1e-7)
            disparity_prop = propagate_all_frames(depth, disparity_opt, disparity_opt_back)
            print("==========", disparity_prop.min(), disparity_prop.max(), disparity_prop.shape, depth.shape)
            save_depth_video(disparity_prop.squeeze(0), os.path.join("tmp", prompt_str + "_depth_prop_refine.mp4"), fps=10)
        elif propagation == "depth_anything":
            # depth = compute_initial_depth(batch_images[7:8]) * 255
            # disparity_prop = propagate_disparity_warping_precompute(depth / 255, disparity_opt, disparity_opt_back, init_index=7)
            # depth = compute_initial_depth(torch.cat([batch_images[2:3], batch_images[7:8], batch_images[12:13]])) * 255
            # disparity_prop_a = propagate_disparity_warping_precompute(depth[0:1] / 255, disparity_opt[:5], disparity_opt_back[:5], init_index=2)
            # disparity_prop_b = propagate_disparity_warping_precompute(depth[1:2] / 255, disparity_opt[5:10], disparity_opt_back[5:10], init_index=2)
            # disparity_prop_c = propagate_disparity_warping_precompute(depth[2:3] / 255, disparity_opt[10:], disparity_opt_back[10:], init_index=2)
            # disparity_prop_b[:, 0:1] = (disparity_prop_b[:, 0:1] + disparity_prop_a[:, 5:6]) / 2
            # disparity_prop_c[:, 0:1] = (disparity_prop_c[:, 0:1] + disparity_prop_b[:, 5:6]) / 2
            # disparity_prop = torch.cat([disparity_prop_a[:, :5], disparity_prop_b[:, :5], disparity_prop_c], dim=1)
            depth = compute_initial_depth(batch_images) * 255
            disparity_prop = propagate_all_frames(depth, disparity_opt, disparity_opt_back)
            # depth_vid = model.infer_frames(frames.numpy(), depth_vid, save_to="tmp/xxx.mp4", num_denoising_steps_dissovling=50, dissolve_step=40)
            save_depth_video(disparity_prop, os.path.join("tmp", prompt_str + "_depth_prop_refine.mp4"), fps=10)
        elif propagation == "depthcrafter":
            disparity_prop = compute_depthcrafter(
                batch_images,
                os.path.join("tmp", prompt_str + ".mp4"),
                num_denoising_steps_dissovling=50,
                dissolve_step=50 - dissolve_step,
            )[None] ** scale_disparity_factor
        elif propagation == "video_depthanything":
            disparity_prop = compute_video_depth_anything(
                batch_images, os.path.join("tmp", prompt_str + ".mp4"))[None] ** scale_disparity_factor
        else:
            raise NotImplementedError(propagation)
        print(disparity_prop.min(), disparity_prop.max(), disparity_prop.max(), disparity_prop.shape, "-----------------------------")

        editor = BNAttention(start_step=stereo_att_start_step, end_step=stereo_att_end_step, direction='uni')
        regiter_attention_editor_diffusers(model, editor)

        iters = 7
        masked_latents = None
        # fix_xT = False
        masked_latents_list = [] 
        masked_latents_left_list = []
        for i in range(iters):
            # if i % 3 == 0:
            fixed_noise = noise_like((B, 4, T, H // 8, W // 8), batch_images.device, False)
            fixed_noise_2 = noise_like((B, 4, T, H // 8, W // 8), batch_images.device, False)
            seed_everything(seed)
            noisy_restart = True
            reuse_modify_latents = True
            clean_border = False
            warped_cond = False

            x_T = None
            if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                warped_cond = False
                clean_border = True

            if i != 0:
                reuse_modify_latents = False

            # if i != 0:
            #     x_T = intermediate["init_xT"][0][0:1]
            #     # Random generator take one step.
            #     noise_like((B, 4, T, H // 8, W // 8), batch_images.device, False)
            #     warped_cond = False
            #     reuse_modify_latents = False

            # if i == 0 or i > iters - 2:
            # if (i) == 1:
            #     import random
            #     seed_everything(random.randint(0, 1000000000))
            # else:
            #     seed_everything(seed)

            # if i in [1, 3, 5]:
            #     import random
            #     noisy_restart = True
            #     seed_everything(random.randint(0, 1000000000))
            if i == iters - 1:
                seed_everything(seed)
                noisy_restart = False
                # clean_border = False

            print(f"Iter {i + 1}/{iters}, noisy_restart={i != iters - 1}")
            # modify_step = 5 if i == 0 else 6
            modify_step = [0, 1, 2, 3, 4, 5]
            # modify_step = [0, 1, 2, 3, 4, 5, 6, 7]
            # modify_step = on_step(i)
            # x_T = intermediate["init_xT"][0][0:1] if fix_xT and i != 0 else None
            samples, intermediate = ddim_stereo_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=noise_shape[0],
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            x_T=x_T,
                                            unconditional_guidance_scale=cfg_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            temporal_length=noise_shape[2],
                                            conditional_guidance_scale_temporal=temporal_cfg_scale,
                                            fs=fs,
                                            timestep_spacing=timestep_spacing,
                                            guidance_rescale=guidance_rescale,
                                            disparity=disparity_prop,
                                            is_duplicate=True,
                                            stereo_scale_factor=stereo_scale_factor,
                                            masked_latents=masked_latents,
                                            is_final=False,
                                            clean_border=clean_border,
                                            cur_iter=i,
                                            # fft=True,
                                            total_iter=iters,
                                            modify_step=modify_step,
                                            # noisy_restart=i != iters - 1,
                                            noisy_restart=noisy_restart,
                                            fixed_noise=fixed_noise,
                                            fixed_noise_2=fixed_noise_2,
                                            # noisy_restart_2=True,
                                            # output_step=None if i != iters - 1 else 8,
                                            imtext_cond_warped=imtext_cond_warped,
                                            img_tensor_repeat_warped=img_tensor_repeat_warped,
                                            warped_cond=warped_cond,
                                            reuse_modify_latents=reuse_modify_latents,
                                            **kwargs
                                            )
            masked_latents = intermediate["masked_latents"]
            masked_latents_left = intermediate["masked_latents_left"]
            # if i != iters - 1:
            #     masked_latents[0] = masked_latents[0] + torch.randn(masked_latents[0])
            # masked_latents_list.append(masked_latents[0].detach())
            # masked_latents_left_list.append(masked_latents_left[0].detach())
            mask_r = intermediate["mask_r"]

        masked_latent_all = masked_latents[-1]
        masked_latents_left_all = masked_latents_left[-1]
        # if propagation:
        #     olist = [(6, 8), (10, 8), (15, 6), (20, 6), (25, 4), (30, 4), (35, 2), (40, 2)]
        # else:
        # olist = [(6, 4), (10, 3), (15, 2), (20, 1), (25, 1), (30, 1), (35, 1), (40, 1)]
        # olist = [(6, 8), (10, 8), (15, 8), (20, 8), (25, 8), (30, 6), (35, 6), (40, 6), (45, 6)]
        # olist = [(6, 8), (10, 8), (15, 8), (20, 8), (25, 8), (30, 6), (35, 6), (36, 2), (37, 2), (38, 2), (39, 2), (40, 6), (45, 6)]
        # olist = [(6, 8), (10, 8), (15, 8), (20, 6), (25, 6), (30, 6), (35, 5), (40, 4), (45, 3)]
        # olist = [(6, 8), (9, 8), (14, 8), (19, 6), (24, 6), (29, 6), (34, 6), (39, 6), (44, 6), (49, 2)]
        olist = [
            (6, 8), (9, 8),
            # (10, 3), (11, 3), (12, 3), (13, 3),
            (14, 3),
            # (15, 3), (16, 3), (17, 3), (18, 3),
            (19, 6),
            # (20, 3), (21, 3), (22, 3), (23, 3),
            (24, 6),
            # (25, 3), (26, 3), (27, 3), (28, 3),
            (29, 6),
            # (30, 3), (31, 3), (32, 3), (33, 3),
            (34, 6),
            # (35, 3), (36, 3), (37, 3), (38, 3),
            (39, 6), (44, 4), (49, 4)
        ]
        # olist = [(10 + i, 3) for i in range(15)] + [(30, 6), (35, 5), (40, 4)]
        # NOTE: for 512
        # olist = [(6, 8), (10, 8), (15, 8), (20, 6), (25, 6), (30, 6), (35, 6), (40, 6), (45, 6)]
        # olist = [(6, 4), (10, 4), (15, 4), (20, 3), (25, 3), (30, 3), (35, 3), (40, 3), (45, 3)]
        # NOTE: for 1024
        # olist = [(6, 9), (10, 8), (15, 7), (20, 6), (25, 5), (30, 4), (35, 4), (40, 4), (45, 4)]
        # olist = [(6, 4), (10, 4), (15, 4), (20, 3), (25, 3), (30, 3), (35, 2), (40, 1)]
        # olist = [(6, 4), (10, 4), (15, 4), (20, 4), (25, 4), (30, 4), (35, 2), (40, 2), (45, 2)]
        # olist = [(6, 9), (10, 8), (15, 7), (20, 9), (25, 9), (30, 9), (35, 9), (40, 9), (45, 9)]

        # olist = [(6, 3), (10, 3), (15, 3), (20, 3), (25, 3), (30, 3), (35, 3), (40, 3), (45, 3)]
        warp_steps_all = [a[0] for a in olist]
        if not refine_non_mask:
            olist = [(i, j) for i, j in olist if i <= max_warp_step]
        else:
            olist = [(i, j) for i, j in olist if i <= max_warp_step] + [(i, 2) for i, j in olist if i > max_warp_step and i != 49]

        warp_steps_a = [i for i in warp_steps_all if i <= max_warp_step] + [max_warp_step]  # If max_warp_step is not in the list
        warp_steps = sorted(list(set(warp_steps_a)))
        if len(warp_steps) != warp_steps_a and len(olist) != 0:
            olist += [(max_warp_step, olist[-1][-1])]
        # if refine_non_mask:
        #     olist = list(warp_steps) + list(refine_non_mask)

        print(olist, warp_steps)

        seed_everything(seed)
        fixed_noise = noise_like((B, 4, T, H // 8, W // 8), batch_images.device, False)
        fixed_noise_2 = noise_like((B, 4, T, H // 8, W // 8), batch_images.device, False)
        seed_everything(seed)
        samples, intermediate = ddim_stereo_sampler.sample(S=ddim_steps,
                                        conditioning=cond,
                                        batch_size=noise_shape[0],
                                        shape=noise_shape[1:],
                                        verbose=False,
                                        x_T=x_T,
                                        unconditional_guidance_scale=cfg_scale,
                                        unconditional_conditioning=uc,
                                        eta=ddim_eta,
                                        temporal_length=noise_shape[2],
                                        conditional_guidance_scale_temporal=temporal_cfg_scale,
                                        fs=fs,
                                        timestep_spacing=timestep_spacing,
                                        guidance_rescale=guidance_rescale,
                                        disparity=disparity_prop,
                                        is_duplicate=True,
                                        stereo_scale_factor=stereo_scale_factor,
                                        is_final=True,
                                        mimic_final=True,
                                        # masked_latents=masked_latents,
                                        step_and_latent=(
                                            modify_step[-1],
                                            torch.cat([masked_latents_left_all, masked_latent_all]),
                                            mask_r[0]
                                        ),
                                        modify_step=warp_steps,
                                        others=intermediate["others"][-1],
                                        # modify_step=it,
                                        # stereo_repeat_times=k,
                                        repeat_steps=olist,
                                        # modify_step=[*modify_step[2:]],
                                        # total_iter=iters,
                                        warped_cond = False,
                                        clean_border = True,
                                        fixed_noise=fixed_noise,
                                        fixed_noise_2=fixed_noise_2,
                                        imtext_cond_warped=imtext_cond_warped,
                                        img_tensor_repeat_warped=img_tensor_repeat_warped,
                                        left_latents=left_latents,
                                        propagation=propagation,
                                        max_warp_step=max_warp_step,
                                        **kwargs
                                        )

        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)

    ## batch, <samples>, c, t, h, w
    batch_variants = torch.stack(batch_variants, dim=1)
    return batch_variants
