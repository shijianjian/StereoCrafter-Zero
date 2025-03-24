from typing import Union, Optional, List, Dict, Callable

import sys
sys.path.append(".")
sys.path.append("./DepthCrafter")
import os
import numpy as np
import torch
from diffusers.training_utils import set_seed
import torchvision
import kornia
from diffusers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _resize_with_antialiasing,
    StableVideoDiffusionPipelineOutput,
    StableVideoDiffusionPipeline,
    retrieve_timesteps,
)

from DepthCrafter.depthcrafter.depth_crafter_ppl import DepthCrafterPipeline as _DepthCrafterPipeline
from DepthCrafter.depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from DepthCrafter.depthcrafter.utils import vis_sequence_depth, save_video, read_video_frames


class DepthCrafterPipeline(_DepthCrafterPipeline):

    @torch.no_grad()
    def forward_and_dissolve(
        self,
        video: Union[np.ndarray, torch.Tensor],
        inference_step: int,
        num_denoising_steps_dissovling: int,
        height: int = 576,
        width: int = 1024,
        num_inference_steps: int = 1,
        guidance_scale: float = 1.0,
        window_size: Optional[int] = 110,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        overlap: int = 25,
        track_time: bool = False,
    ):
        """
        :param video: in shape [t, h, w, c] if np.ndarray or [t, c, h, w] if torch.Tensor, in range [0, 1]
        :param height:
        :param width:
        :param num_inference_steps:
        :param guidance_scale:
        :param window_size: sliding window processing size
        :param fps:
        :param motion_bucket_id:
        :param noise_aug_strength:
        :param decode_chunk_size:
        :param generator:
        :param latents:
        :param output_type:
        :param callback_on_step_end:
        :param callback_on_step_end_tensor_inputs:
        :param return_dict:
        :return:
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        num_frames = video.shape[0]
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else 8
        if num_frames <= window_size:
            window_size = num_frames
            overlap = 0
        stride = window_size - overlap

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(video, height, width)

        # 2. Define call parameters
        batch_size = 1
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = guidance_scale

        # 3. Encode input video
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video.transpose(0, 3, 1, 2))
        else:
            assert isinstance(video, torch.Tensor)
        video = video.to(device=device, dtype=self.dtype)
        video = video * 2.0 - 1.0  # [0,1] -> [-1,1], in [t, c, h, w]

        if track_time:
            start_event = torch.cuda.Event(enable_timing=True)
            encode_event = torch.cuda.Event(enable_timing=True)
            denoise_event = torch.cuda.Event(enable_timing=True)
            decode_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        video_embeddings = self.encode_video(
            video, chunk_size=decode_chunk_size
        ).unsqueeze(
            0
        )  # [1, t, 1024]
        torch.cuda.empty_cache()
        # 4. Encode input image using VAE
        noise = randn_tensor(
            video.shape, generator=generator, device=device, dtype=video.dtype
        )
        video = video + noise_aug_strength * noise  # in [t, c, h, w]

        # pdb.set_trace()
        needs_upcasting = (
            self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        )
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        video_latents = self.encode_vae_video(
            video.to(self.vae.dtype),
            chunk_size=decode_chunk_size,
        ).unsqueeze(
            0
        )  # [1, t, c, h, w]

        if track_time:
            encode_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(encode_event)
            print(f"Elapsed time for encoding video: {elapsed_time_ms} ms")

        torch.cuda.empty_cache()

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            7,
            127,
            noise_aug_strength,
            video_embeddings.dtype,
            batch_size,
            1,
            False,
        )  # [1 or 2, 3]
        added_time_ids = added_time_ids.to(device)

        # 6. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, None, None
        )
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        # 7. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents_init = self.prepare_latents(
            batch_size,
            window_size,
            num_channels_latents,
            height,
            width,
            video_embeddings.dtype,
            device,
            generator,
            latents,
        )  # [1, t, c, h, w]
        latents_all = None

        idx_start = 0
        if overlap > 0:
            weights = torch.linspace(0, 1, overlap, device=device)
            weights = weights.view(1, overlap, 1, 1, 1)
        else:
            weights = None

        torch.cuda.empty_cache()

        # inference strategy for long videos
        # two main strategies: 1. noise init from previous frame, 2. segments stitching
        while idx_start < num_frames - overlap:
            idx_end = min(idx_start + window_size, num_frames)
            self.scheduler.set_timesteps(num_inference_steps, device=device)

            # 9. Denoising loop
            latents = latents_init[:, : idx_end - idx_start].clone()
            latents_init = torch.cat(
                [latents_init[:, -overlap:], latents_init[:, :stride]], dim=1
            )

            video_latents_current = video_latents[:, idx_start:idx_end]
            video_embeddings_current = video_embeddings[:, idx_start:idx_end]

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if latents_all is not None and i == 0:
                        latents[:, :overlap] = (
                            latents_all[:, -overlap:]
                            + latents[:, :overlap]
                            / self.scheduler.init_noise_sigma
                            * self.scheduler.sigmas[i]
                        )

                    latent_model_input = latents  # [1, t, c, h, w]
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )  # [1, t, c, h, w]
                    latent_model_input = torch.cat(
                        [latent_model_input, video_latents_current], dim=2
                    )
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=video_embeddings_current,
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]
                    # perform guidance
                    if self.do_classifier_free_guidance:
                        latent_model_input = latents
                        latent_model_input = self.scheduler.scale_model_input(
                            latent_model_input, t
                        )
                        latent_model_input = torch.cat(
                            [latent_model_input, torch.zeros_like(latent_model_input)],
                            dim=2,
                        )
                        noise_pred_uncond = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=torch.zeros_like(
                                video_embeddings_current
                            ),
                            added_time_ids=added_time_ids,
                            return_dict=False,
                        )[0]

                        noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred - noise_pred_uncond
                        )
                    latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(
                            self, i, t, callback_kwargs
                        )

                        latents = callback_outputs.pop("latents", latents)

                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps
                        and (i + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()

            if inference_step is not None:
                latent_model_input = latents  # [1, t, c, h, w]
                scheduler_old = self.scheduler
                # self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
                self.scheduler.set_timesteps(num_denoising_steps_dissovling, device=device)
                self.scheduler._step_index = inference_step
                timesteps, _ = retrieve_timesteps(
                    self.scheduler, num_denoising_steps_dissovling, device, None, None
                )

                t = timesteps[inference_step]
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )  # [1, t, c, h, w]
                latent_model_input = torch.cat(
                    [latent_model_input, video_latents_current], dim=2
                )
                print(t)
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=video_embeddings_current,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]
                latents = self.scheduler.step(noise_pred, t, latents).pred_original_sample.to(noise_pred)
                # latents = self.predict_start_from_noise(noise_pred, t, latents)
                self.scheduler = scheduler_old

            if latents_all is None:
                latents_all = latents.clone()
            else:
                assert weights is not None
                # latents_all[:, -overlap:] = (
                #     latents[:, :overlap] + latents_all[:, -overlap:]
                # ) / 2.0
                latents_all[:, -overlap:] = latents[
                    :, :overlap
                ] * weights + latents_all[:, -overlap:] * (1 - weights)
                latents_all = torch.cat([latents_all, latents[:, overlap:]], dim=1)

            idx_start += stride

        if track_time:
            denoise_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = encode_event.elapsed_time(denoise_event)
            print(f"Elapsed time for denoising video: {elapsed_time_ms} ms")

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)

            frames = self.decode_latents(latents_all, num_frames, decode_chunk_size)

            if track_time:
                decode_event.record()
                torch.cuda.synchronize()
                elapsed_time_ms = denoise_event.elapsed_time(decode_event)
                print(f"Elapsed time for decoding video: {elapsed_time_ms} ms")

            frames = self.video_processor.postprocess_video(
                video=frames, output_type=output_type
            )
        else:
            frames = latents_all

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)

    @torch.no_grad()
    def one_step_dissolve(
        self,
        video: Union[np.ndarray, torch.Tensor],
        depth_video: Union[np.ndarray, torch.Tensor], 
        inference_step: int,
        num_denoising_steps_dissovling: int,
        height: int = 576,
        width: int = 1024,
        guidance_scale: float = 1.0,
        noise_aug_strength: float = 0.02,
        window_size: Optional[int] = 110,
        decode_chunk_size: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        overlap: int = 25,
        track_time: bool = False,
    ):
        """
        :param video: in shape [t, h, w, c] if np.ndarray or [t, c, h, w] if torch.Tensor, in range [0, 1]
        :param height:
        :param width:
        :param num_inference_steps:
        :param guidance_scale:
        :param window_size: sliding window processing size
        :param fps:
        :param motion_bucket_id:
        :param noise_aug_strength:
        :param decode_chunk_size:
        :param generator:
        :param latents:
        :param output_type:
        :param callback_on_step_end:
        :param callback_on_step_end_tensor_inputs:
        :param return_dict:
        :return:
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        num_frames = video.shape[0]
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else 8
        if num_frames <= window_size:
            window_size = num_frames
            overlap = 0
        stride = window_size - overlap

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(video, height, width)

        # 2. Define call parameters
        batch_size = 1
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = guidance_scale

        # 3. Encode input video
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video.transpose(0, 3, 1, 2))
        else:
            assert isinstance(video, torch.Tensor)
        video = video.to(device=device, dtype=self.dtype)
        video = video * 2.0 - 1.0  # [0,1] -> [-1,1], in [t, c, h, w]

        # 3. Encode depth video
        if isinstance(depth_video, np.ndarray):
            depth_video = torch.from_numpy(depth_video.transpose(0, 3, 1, 2))
        else:
            assert isinstance(depth_video, torch.Tensor)
        depth_video = depth_video.to(device=device, dtype=self.dtype)
        depth_video = depth_video * 2.0 - 1.0  # [0,1] -> [-1,1], in [t, c, h, w]

        if track_time:
            start_event = torch.cuda.Event(enable_timing=True)
            encode_event = torch.cuda.Event(enable_timing=True)
            denoise_event = torch.cuda.Event(enable_timing=True)
            decode_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        video_embeddings = self.encode_video(
            video, chunk_size=decode_chunk_size
        ).unsqueeze(
            0
        )  # [1, t, 1024]
        torch.cuda.empty_cache()
        # 4. Encode input image using VAE
        noise = randn_tensor(
            video.shape, generator=generator, device=device, dtype=video.dtype
        )
        video = video + noise_aug_strength * noise  # in [t, c, h, w]

        torch.cuda.empty_cache()
        # 4. Encode input image using VAE

        # pdb.set_trace()
        needs_upcasting = (
            self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        )
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        video_latents = self.encode_vae_video(
            video.to(self.vae.dtype),
            chunk_size=decode_chunk_size,
        ).unsqueeze(
            0
        )  # [1, t, c, h, w]

        depth_video_latents = self.encode_vae_video(
            depth_video.to(self.vae.dtype),
            chunk_size=decode_chunk_size,
        ).unsqueeze(
            0
        )  # [1, t, c, h, w]

        if track_time:
            encode_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(encode_event)
            print(f"Elapsed time for encoding video: {elapsed_time_ms} ms")

        torch.cuda.empty_cache()

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            7,
            127,
            noise_aug_strength,
            video_embeddings.dtype,
            batch_size,
            1,
            False,
        )  # [1 or 2, 3]
        added_time_ids = added_time_ids.to(device)

        # 6. Prepare timesteps
        # timesteps, num_inference_steps = retrieve_timesteps(
        #     self.scheduler, num_inference_steps, device, None, None
        # )
        # num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        # self._num_timesteps = len(timesteps)

        # 7. Prepare latent variables
        # num_channels_latents = self.unet.config.in_channels
        # latents_init = self.prepare_latents(
        #     batch_size,
        #     window_size,
        #     num_channels_latents,
        #     height,
        #     width,
        #     video_embeddings.dtype,
        #     device,
        #     generator,
        #     latents,
        # )  # [1, t, c, h, w]
        latents_all = None

        idx_start = 0
        if overlap > 0:
            weights = torch.linspace(0, 1, overlap, device=device)
            weights = weights.view(1, overlap, 1, 1, 1)
        else:
            weights = None

        torch.cuda.empty_cache()

        # # inference strategy for long videos
        # # two main strategies: 1. noise init from previous frame, 2. segments stitching
        # while idx_start < num_frames - overlap:
        #     idx_end = min(idx_start + window_size, num_frames)
        #     self.scheduler.set_timesteps(num_inference_steps, device=device)

        #     # 9. Denoising loop
        #     latents = latents_init[:, : idx_end - idx_start].clone()
        #     latents_init = torch.cat(
        #         [latents_init[:, -overlap:], latents_init[:, :stride]], dim=1
        #     )

        #     video_latents_current = video_latents[:, idx_start:idx_end]
        #     video_embeddings_current = video_embeddings[:, idx_start:idx_end]

        #     with self.progress_bar(total=num_inference_steps) as progress_bar:
        #         for i, t in enumerate(timesteps):
        #             if latents_all is not None and i == 0:
        #                 latents[:, :overlap] = (
        #                     latents_all[:, -overlap:]
        #                     + latents[:, :overlap]
        #                     / self.scheduler.init_noise_sigma
        #                     * self.scheduler.sigmas[i]
        #                 )

        #             latent_model_input = latents  # [1, t, c, h, w]
        #             latent_model_input = self.scheduler.scale_model_input(
        #                 latent_model_input, t
        #             )  # [1, t, c, h, w]
        #             latent_model_input = torch.cat(
        #                 [latent_model_input, video_latents_current], dim=2
        #             )
        #             noise_pred = self.unet(
        #                 latent_model_input,
        #                 t,
        #                 encoder_hidden_states=video_embeddings_current,
        #                 added_time_ids=added_time_ids,
        #                 return_dict=False,
        #             )[0]
        #             # perform guidance
        #             if self.do_classifier_free_guidance:
        #                 latent_model_input = latents
        #                 latent_model_input = self.scheduler.scale_model_input(
        #                     latent_model_input, t
        #                 )
        #                 latent_model_input = torch.cat(
        #                     [latent_model_input, torch.zeros_like(latent_model_input)],
        #                     dim=2,
        #                 )
        #                 noise_pred_uncond = self.unet(
        #                     latent_model_input,
        #                     t,
        #                     encoder_hidden_states=torch.zeros_like(
        #                         video_embeddings_current
        #                     ),
        #                     added_time_ids=added_time_ids,
        #                     return_dict=False,
        #                 )[0]

        #                 noise_pred = noise_pred_uncond + self.guidance_scale * (
        #                     noise_pred - noise_pred_uncond
        #                 )
        #             latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        #             if callback_on_step_end is not None:
        #                 callback_kwargs = {}
        #                 for k in callback_on_step_end_tensor_inputs:
        #                     callback_kwargs[k] = locals()[k]
        #                 callback_outputs = callback_on_step_end(
        #                     self, i, t, callback_kwargs
        #                 )

        #                 latents = callback_outputs.pop("latents", latents)

        #             if i == len(timesteps) - 1 or (
        #                 (i + 1) > num_warmup_steps
        #                 and (i + 1) % self.scheduler.order == 0
        #             ):
        #                 progress_bar.update()

        while idx_start < num_frames - overlap:
            idx_end = min(idx_start + window_size, num_frames)
            video_latents_current = video_latents[:, idx_start:idx_end]
            depth_video_latents_current = depth_video_latents[:, idx_start:idx_end]
            video_embeddings_current = video_embeddings[:, idx_start:idx_end]

            # latents = torch.cat(
            #     [video_latents_current, depth_video_latents_current], dim=2
            # )
            latents = depth_video_latents_current

            if inference_step is not None:
                latent_model_input = latents  # [1, t, c, h, w]
                scheduler_old = self.scheduler
                # self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
                self.scheduler.set_timesteps(num_denoising_steps_dissovling, device=device)
                self.scheduler._step_index = inference_step
                timesteps, _ = retrieve_timesteps(
                    self.scheduler, num_denoising_steps_dissovling, device, None, None
                )
                t = timesteps[inference_step]

                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )  # [1, t, c, h, w]
                latent_model_input = torch.cat(
                    [latent_model_input, video_latents_current], dim=2
                )
                print(t)
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=video_embeddings_current,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]
                latents = self.scheduler.step(noise_pred, t, latents).pred_original_sample.to(noise_pred)
                # latents = self.predict_start_from_noise(noise_pred, t, latents)
                self.scheduler = scheduler_old

            if latents_all is None:
                latents_all = latents.clone()
            else:
                assert weights is not None
                # latents_all[:, -overlap:] = (
                #     latents[:, :overlap] + latents_all[:, -overlap:]
                # ) / 2.0
                latents_all[:, -overlap:] = latents[
                    :, :overlap
                ] * weights + latents_all[:, -overlap:] * (1 - weights)
                latents_all = torch.cat([latents_all, latents[:, overlap:]], dim=1)

            idx_start += stride

        if track_time:
            denoise_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = encode_event.elapsed_time(denoise_event)
            print(f"Elapsed time for denoising video: {elapsed_time_ms} ms")

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)

            frames = self.decode_latents(latents_all, num_frames, decode_chunk_size)

            if track_time:
                decode_event.record()
                torch.cuda.synchronize()
                elapsed_time_ms = denoise_event.elapsed_time(decode_event)
                print(f"Elapsed time for decoding video: {elapsed_time_ms} ms")

            frames = self.video_processor.postprocess_video(
                video=frames, output_type=output_type
            )
        else:
            frames = latents_all

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)


    def predict_start_from_noise(self, noise_pred, timestep: int, latent):
        return (
            torch.sqrt(1.0 / self.scheduler.alphas_cumprod[timestep]) * latent
            - torch.sqrt(1.0 / self.scheduler.alphas_cumprod[timestep] - 1) * noise_pred
        )


class DepthCrafterWrapper:
    def __init__(
        self,
        unet_path: str,
        pre_train_path: str,
        cpu_offload: str = "model",
    ):
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        # load weights of other components from the provided checkpoint
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_train_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # for saving memory, we can offload the model to CPU, or even run the model sequentially to save more memory
        if cpu_offload is not None:
            if cpu_offload == "sequential":
                # This will slow, but save more memory
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
        else:
            self.pipe.to("cuda")
        # enable attention slicing and xformers memory efficient attention
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(e)
            print("Xformers is not enabled")
        self.pipe.enable_attention_slicing()

    def infer(
        self,
        video: str,
        num_denoising_steps: int,
        guidance_scale: float,
        save_folder: str = "./demo_output",
        window_size: int = 110,
        process_length: int = 195,
        overlap: int = 25,
        max_res: int = 1024,
        dataset: str = "open",
        target_fps: int = 15,
        seed: int = 42,
        track_time: bool = True,
        save_npz: bool = False,
    ):
        set_seed(seed)

        frames, target_fps = read_video_frames(
            video, process_length, target_fps, max_res, dataset,
        )
        print(f"==> video name: {video}, frames shape: {frames.shape}")

        # inference the depth map using the DepthCrafter pipeline
        res = self.infer_frames(
            frames,
            num_denoising_steps=num_denoising_steps,
            guidance_scale=guidance_scale,
            window_size=window_size,
            overlap=overlap,
            seed=seed,
            track_time=track_time,
            output_type="np",
        )
        # with torch.inference_mode():
        #     res = self.pipe(
        #         frames,
        #         height=frames.shape[1],
        #         width=frames.shape[2],
        #         output_type="np",
        #         guidance_scale=guidance_scale,
        #         num_inference_steps=num_denoising_steps,
        #         window_size=window_size,
        #         overlap=overlap,
        #         track_time=track_time,
        #     ).frames[0]
        # # convert the three-channel output to a single channel depth map
        # res = res.sum(-1) / res.shape[-1]
        # normalize the depth map to [0, 1] across the whole video
        res = (res - res.min()) / (res.max() - res.min())
        # visualize the depth map and save the results
        vis = vis_sequence_depth(res)
        # save the depth map and visualization with the target FPS
        save_path = os.path.join(
            save_folder, os.path.splitext(os.path.basename(video))[0]
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_npz:
            np.savez_compressed(save_path + ".npz", depth=res)
        save_video(res, save_path + "_depth.mp4", fps=target_fps)
        save_video(vis, save_path + "_vis.mp4", fps=target_fps)
        save_video(frames, save_path + "_input.mp4", fps=target_fps)
        return [
            save_path + "_input.mp4",
            save_path + "_vis.mp4",
            save_path + "_depth.mp4",
        ]

    def infer_frames(
        self,
        frames: torch.Tensor | np.ndarray,
        depth_frames: torch.Tensor | np.ndarray | None = None,
        num_denoising_steps: int = 25,
        guidance_scale: float = 1.2,
        window_size: int = 110,
        overlap: int = 25,
        seed: int = 42,
        track_time: bool = True,
        output_type="np",
        save_to: str | None = None,
        target_fps: int = 10,
        dissolve_step: int | None = None,
        num_denoising_steps_dissovling: int = 50,
    ):
        set_seed(seed)

        if dissolve_step is None:
            # inference the depth map using the DepthCrafter pipeline
            with torch.inference_mode():
                res = self.pipe(
                    frames,
                    height=frames.shape[1],
                    width=frames.shape[2],
                    output_type=output_type,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_denoising_steps,
                    window_size=window_size,
                    overlap=overlap,
                    track_time=track_time,
                ).frames[0]
        elif depth_frames is None:
            with torch.inference_mode():
                res = self.pipe.forward_and_dissolve(
                    frames,
                    inference_step=dissolve_step,
                    num_denoising_steps_dissovling=num_denoising_steps_dissovling,
                    height=frames.shape[1],
                    width=frames.shape[2],
                    output_type=output_type,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_denoising_steps,
                    window_size=window_size,
                    overlap=overlap,
                    track_time=track_time,
                ).frames[0]
        else:
            with torch.inference_mode():
                res = self.pipe.one_step_dissolve(
                    frames,
                    depth_frames,
                    inference_step=dissolve_step,
                    num_denoising_steps_dissovling=num_denoising_steps_dissovling,
                    height=frames.shape[1],
                    width=frames.shape[2],
                    output_type=output_type,
                    guidance_scale=guidance_scale,
                    window_size=window_size,
                    overlap=overlap,
                    track_time=track_time,
                ).frames[0]

        # convert the three-channel output to a single channel depth map
        res = res.sum(-1) / res.shape[-1]

        if save_to:
            res_save = (res - res.min()) / (res.max() - res.min())
            # visualize the depth map and save the results
            vis = vis_sequence_depth(res_save)
            # save the depth map and visualization with the target FPS
            self.save_video(torch.tensor(res_save[:, None]), save_to.replace(".mp4", "_depth.mp4"), fps=target_fps)
            self.save_video(torch.tensor(vis).permute(0, 3, 1, 2), save_to.replace(".mp4", "_vis.mp4"), fps=target_fps)

        return res

    def save_video(self, vid_tensor, savepath, fps):
        print(vid_tensor.shape, savepath)
        video = vid_tensor.detach().cpu() # stack in temporal dim [t, 1, h, w]
        if video.shape[1] == 1:
            video = kornia.color.grayscale_to_rgb(video)
        grid = (video * 255).to(torch.uint8).permute(0, 2, 3, 1)
        torchvision.io.write_video(savepath, grid, fps=fps, video_codec='h264', options={'crf': '10'})


if __name__ == "__main__":
    model = DepthCrafterWrapper(
        unet_path="tencent/DepthCrafter",
        pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
        cpu_offload="model",
    )
    import torchvision
    frames, _, _ = torchvision.io.read_video("/ibex/ai/home/shij0c/git/makeit3d/DynamiCrafter/ab.mp4")
    depth_vid = model.infer_frames(frames.numpy(), save_to="tmp/xxx2.mp4", num_denoising_steps=5)
    depth_vid = torch.tensor(depth_vid[:, None]).repeat(1, 3, 1, 1)
    print(depth_vid.min(), depth_vid.max(), depth_vid.shape, frames.shape)
    depth_vid = model.infer_frames(frames.numpy(), depth_vid, save_to="tmp/xxx.mp4", num_denoising_steps_dissovling=50, dissolve_step=40)
    # print(model.infer_frames(frames.numpy(), save_to="tmp/xxx.mp4").shape)
    # print(model.infer_frames(frames.numpy(), save_to="tmp/xxx.mp4", num_denoising_steps=5, num_denoising_steps_dissovling=50, dissolve_step=35).shape)
