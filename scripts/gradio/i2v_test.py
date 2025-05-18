import os
import time
from omegaconf import OmegaConf
import torch
from scripts.evaluation.funcs import load_model_checkpoint, merge_video_sbs, save_videos, batch_ddim_sampling, get_latent_z
from scripts.evaluation.funcs_stereo import batch_stereo_ddim_sampling
from utils.utils import instantiate_from_config
from huggingface_hub import hf_hub_download
from einops import repeat
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything

from PIL import Image


class Image2Video():
    def __init__(self,result_dir='./tmp/',gpu_num=1,resolution='256_256', interp=False) -> None:
        self.resolution = (int(resolution.split('_')[0]), int(resolution.split('_')[1])) #hw
        self.download_model()

        self.result_dir = result_dir
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        
        if interp:
            ckpt_path='checkpoints/dynamicrafter_'+resolution.split('_')[1]+'_interp_v1/model.ckpt'
            config_file='configs/inference_'+resolution.split('_')[1]+'_v1.0.yaml'
        else:
            ckpt_path='checkpoints/dynamicrafter_'+resolution.split('_')[1]+'_v1/model.ckpt'
            config_file='configs/inference_'+resolution.split('_')[1]+'_v1.0.yaml'

        config = OmegaConf.load(config_file)
        model_config = config.pop("model", OmegaConf.create())
        model_config['params']['unet_config']['params']['use_checkpoint']=False   
        model_list = []
        for gpu_id in range(gpu_num):
            model = instantiate_from_config(model_config)
            # model = model.cuda(gpu_id)
            assert os.path.exists(ckpt_path), "Error: checkpoint Not Found!"
            model = load_model_checkpoint(model, ckpt_path)
            model.eval()
            model_list.append(model)
        self.model_list = model_list
        self.save_fps = 8

    def get_image(
        self, image, prompt, steps=50, cfg_scale=7.5, eta=1.0, fs=3, seed=123,
        is_stereo=True, stereo_att_start_step=25, stereo_att_end_step=45, stereo_scale_factor=8,
        max_warp_step=40, image2=None, looped=False, refine_non_mask=False, propagation=False,
        dissolve_step=None,
    ):
        seed_everything(seed)
        transform = transforms.Compose([
            transforms.Resize(min(self.resolution)),
            transforms.CenterCrop(self.resolution),
            ])
        torch.cuda.empty_cache()
        print('start:', prompt, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        start = time.time()
        gpu_id = 0
        if steps > 60:
            steps = 60 
        model = self.model_list[gpu_id]
        model = model.cuda()

        batch_size = 1
        channels = model.model.diffusion_model.out_channels
        frames = model.temporal_length
        h, w = self.resolution[0] // 8, self.resolution[1] // 8
        noise_shape = [batch_size, channels, frames, h, w]

        # text cond
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_emb = model.get_learned_conditioning([prompt])

            # img cond
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(model.device)

            img_tensor_warped = img_tensor  # Not used redundant arg
            
            img_tensor = (img_tensor / 255. - 0.5) * 2
            img_tensor_warped = (img_tensor_warped / 255. - 0.5) * 2

            image_tensor_resized = transform(img_tensor)  # 3, h, w
            videos = image_tensor_resized.unsqueeze(0)  # bchw
            image_tensor_resized_warped = transform(img_tensor_warped)  # 3, h, w
            videos_warped = image_tensor_resized_warped.unsqueeze(0)  # bchw

            z = get_latent_z(model, videos.unsqueeze(2)) # bc,1,hw
            z_warped = get_latent_z(model, videos_warped.unsqueeze(2)) # bc,1,hw
            
            if image2 is not None:
                img_tensor2 = torch.from_numpy(image2).permute(2, 0, 1).float().to(model.device)
                img_tensor2 = (img_tensor2 / 255. - 0.5) * 2

                image_tensor_resized2 = transform(img_tensor2) #3,h,w
                videos2 = image_tensor_resized2.unsqueeze(0) # bchw
                
                z2 = get_latent_z(model, videos2.unsqueeze(2)) #bc,1,hw

            img_tensor_repeat = repeat(z, 'b c t h w -> b c (repeat t) h w', repeat=frames)
            if image2 is not None or looped:
                img_tensor_repeat = torch.zeros_like(img_tensor_repeat)
                img_tensor_repeat[:,:,:1,:,:] = z
            if image2 is not None:
                img_tensor_repeat[:,:,-1:,:,:] = z2

            img_tensor_repeat_warped = repeat(z_warped, 'b c t h w -> b c (repeat t) h w', repeat=frames)

            cond_images = model.embedder(img_tensor.unsqueeze(0))  ## blc
            cond_images_warped = model.embedder(img_tensor_warped.unsqueeze(0))  ## blc
            img_emb = model.image_proj_model(cond_images)
            img_emb_warped = model.image_proj_model(cond_images_warped)

            imtext_cond = torch.cat([text_emb, img_emb], dim=1)
            imtext_cond_warped = torch.cat([text_emb, img_emb_warped], dim=1)

            fs = torch.tensor([fs], dtype=torch.long, device=model.device)
            cond = {
                "c_crossattn": [imtext_cond], "fs": fs, "c_concat": [img_tensor_repeat],
            }

            ## b,samples,c,t,h,w
            prompt_str = prompt.replace("/", "_slash_") if "/" in prompt else prompt
            prompt_str = prompt_str.replace(" ", "_") if " " in prompt else prompt_str
            prompt_str=prompt_str[:40]
            if len(prompt_str) == 0:
                prompt_str = 'empty_prompt'

            ## inference
            if is_stereo:
                batch_samples = batch_stereo_ddim_sampling(
                    model, cond, noise_shape, n_samples=1, ddim_steps=steps, ddim_eta=eta, cfg_scale=cfg_scale,
                    stereo_att_start_step=stereo_att_start_step, stereo_att_end_step=stereo_att_end_step, stereo_scale_factor=stereo_scale_factor,
                    imtext_cond_warped=imtext_cond_warped, prompt_str=prompt_str, img_tensor_repeat_warped=img_tensor_repeat_warped, max_warp_step=max_warp_step,
                    refine_non_mask=refine_non_mask, seed=seed, propagation=propagation, dissolve_step=dissolve_step,
                )
                ## remove the first and the last frame
                if image2 is not None:
                    batch_samples = batch_samples[:,:,:,1:-1,...]
                if looped:
                    batch_samples = batch_samples[:,:,:,1:,...]
            else:
                batch_samples = batch_ddim_sampling(
                    model, cond, noise_shape, n_samples=1, ddim_steps=steps, ddim_eta=eta, cfg_scale=cfg_scale
                )

        if is_stereo:
            suffix = "-r" if stereo_scale_factor > 0 else "-l"
            batch_left = batch_samples[0]
            batch_right = batch_samples[1]
            overlay = torch.stack([batch_right[:, 1], batch_left[:, 1], batch_left[:, 2]], dim=1)[None]
            save_videos(overlay, self.result_dir, filenames=[prompt_str + suffix + "_overlay"], fps=self.save_fps)
            batch_samples = torch.cat([batch_samples[0:1], batch_samples[1:2]], dim=-1)

        save_videos(batch_samples, self.result_dir, filenames=[prompt_str + suffix], fps=self.save_fps)
        if is_stereo:
            print(f"Saved in {prompt_str + suffix}. Time used: {(time.time() - start):.2f} seconds")
        else:
            print(f"Saved in {prompt_str}. Time used: {(time.time() - start):.2f} seconds")

        # Merge r/l if both exist
        rpath = os.path.join(self.result_dir, prompt_str + "-r.mp4")
        lpath = os.path.join(self.result_dir, prompt_str + "-l.mp4")
        if os.path.exists(rpath) and os.path.exists(lpath):
            merge_video_sbs(lpath, rpath, os.path.join(self.result_dir, prompt_str + "_final"))
        model = model.cpu()
        return os.path.join(self.result_dir, f"{prompt_str}.mp4")

    def download_model(self):
        REPO_ID = 'Doubiiu/DynamiCrafter_'+str(self.resolution[1]) if self.resolution[1]!=256 else 'Doubiiu/DynamiCrafter'
        filename_list = ['model.ckpt']
        if not os.path.exists('./checkpoints/dynamicrafter_'+str(self.resolution[1])+'_v1/'):
            os.makedirs('./checkpoints/dynamicrafter_'+str(self.resolution[1])+'_v1/')
        for filename in filename_list:
            local_file = os.path.join('./checkpoints/dynamicrafter_'+str(self.resolution[1])+'_v1/', filename)
            if not os.path.exists(local_file):
                hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir='./checkpoints/dynamicrafter_'+str(self.resolution[1])+'_v1/', local_dir_use_symlinks=False)


if __name__ == '__main__':

    run_both = False

    # Options for depth models: video_depthanything, depth_pro, depthcrafter, depth_anything
    # Note that the dissolve depth step is only working with depthcrafter model.

    # Note that the last parameter (dissolve step) is more crucial to be tweaked.
    # The last warping step shall be around 34 and 39.

    i2v_examples_1024 = [
        ['prompts/1024/astronaut04.png', 'a man in an astronaut suit playing a guitar', 50, 7.5, 1.0, 6, 123, 30, 50, 4, 39, 1, False, "depthcrafter", 20],
        ['prompts/1024/bloom01.png', 'time-lapse of a blooming flower with leaves and a stem', 50, 7.5, 1.0, 10, 123, 30, 50, 4, 34, 1, False, "depthcrafter", 30],
        ['prompts/1024/girl07.png', 'a beautiful woman with long hair and a dress blowing in the wind', 50, 7.5, 1.0, 30, 123, 10, 50, 4, 34, 1, False, "depthcrafter", 30],
        ['prompts/1024/pour_bear.png', 'pouring beer into a glass of ice and beer', 50, 7.5, 1.0, 10, 123, 50, 50, 4, 34, 1, False, "depthcrafter", 30],
        ['prompts/1024/robot01.png', 'a robot is walking through a destroyed city', 50, 7.5, 1.0, 10, 123, 0, 50, 4, 34, 1, False, "depthcrafter", 5],
        ['prompts/1024/firework03.png', 'fireworks display', 50, 7.5, 1.0, 10, 123, 0, 50, 4, 34, 1, False, "depthcrafter", 5],
    ]

    i2v_examples_512 = [
        ['prompts/512/bloom01.png', 'time-lapse of a blooming flower with leaves and a stem', 50, 7.5, 1.0, 24, 123, 0, 50, 4, 44, 1, False, "depthcrafter", 30],
        ['prompts/512/campfire.png', 'a bonfire is lit in the middle of a field', 50, 7.5, 1.0, 24, 123, 0, 50, 4, 44, 1, False, "depthcrafter", 30],
        ['prompts/512/isometric.png', 'rotating view, small house', 50, 7.5, 1.0, 24, 123, 0, 50, 4, 44, 1, False, "depthcrafter", 30],
        ['prompts/512/girl08.png', 'a woman looking out in the rain', 50, 7.5, 1.0, 24, 1234, 0, 50, 4, 44, 1, False, "depthcrafter"], 30,
        ['prompts/512/zreal_penguin.png', 'a group of penguins walking on a beach', 50, 7.5, 1.0, 20, 123, 0, 50, 4, 44, 2.71828182845904523536, False, "depthcrafter", 30],
        ['prompts/512/ship02.png', 'a sailboat sailing in rough seas with a dramatic sunset', 50, 7.5, 1.0, 24, 123, 0, 50, 8, 34, 1, False, "depthcrafter", 30],
    ]

    i2v_examples_256 = [
        ['prompts/256/art.png', 'man fishing in a boat at sunset', 50, 7.5, 1.0, 3, 234, 0, 50, 4, 45, 1, False, "depthcrafter", 30],
        ['prompts/256/boy.png', 'boy walking on the street', 50, 7.5, 1.0, 3, 125, 0, 50, 4, 45, 1, False, "depthcrafter", 30],
        ['prompts/256/dance1.jpeg', 'two people dancing', 50, 7.5, 1.0, 3, 116, 0, 50, 4, 45, 1, False, "depthcrafter", 30],
        ['prompts/256/fire_and_beach.jpg', 'a campfire on the beach and the ocean waves in the background', 50, 7.5, 1.0, 3, 111, 0, 50, 4, 45, 1, False, "depthcrafter", 30],
        ['prompts/256/girl3.jpeg', 'girl talking and blinking', 50, 7.5, 1.0, 3, 111, 0, 50, 4, 45, 1, False, "depthcrafter", 30],
        ['prompts/256/guitar0.jpeg', 'bear playing guitar happily, snowing', 50, 7.5, 1.0, 3, 122, 0, 50, 4, 45, 1, False, "depthcrafter", 30]
    ]

    i2v_examples_interp_512 = [
        ['prompts/512_interp/smile_01.png', 'a smiling girl', 50, 7.5, 1.0, 5, 12306, 0, 50, 8, 38, 2.71828182845904523536, False, "depthcrafter", 30, 'prompts/512_interp/smile_02.png'],
        ['prompts/512_interp/stone01_01.png', 'rotating view', 50, 7.5, 1.0, 5, 123, 0, 50, 8, 30, 2.71828182845904523536, False, "depthcrafter", 30, 'prompts/512_interp/stone01_02.png'],
        ['prompts/512_interp/walk_01.png', 'man walking', 50, 7.5, 1.0, 5, 345, 0, 50, 8, 35, 2.71828182845904523536, False, "depthcrafter", 30, 'prompts/512_interp/walk_02.png'],
    ]

    i2v_examples_loop_512 = [
        ['prompts/512_loop/24.png', 'a beach with waves and clouds at sunset', 50, 7.5, 1.0, 5, 234, 0, 50, 8, 45, False, "depthcrafter", 30],
        ['prompts/512_loop/36.png', 'clothes swaying in the wind', 50, 7.5, 1.0, 5, 123, 0, 50, 8, 45, 1, False, "depthcrafter", 30],
        ['prompts/512_loop/40.png', 'flowers swaying in the wind', 50, 7.5, 1.0, 5, 234, 0, 50, 8, 45, 1, False, "depthcrafter", 30],
    ]

    # python -m scripts.gradio.i2v_test
    from PIL import Image
    import numpy as np

    i2v, sample_list, is_interp, looped = Image2Video(resolution="576_1024"), i2v_examples_1024, False, False
    # i2v, sample_list, is_interp, looped = Image2Video(resolution="320_512"), i2v_examples_512, False, False
    # i2v, sample_list, is_interp, looped = Image2Video(resolution="320_512", interp=True), i2v_examples_interp_512, True, False
    # i2v, sample_list, is_interp, looped = Image2Video(resolution="320_512", interp=True), i2v_examples_loop_512, True, True
    # i2v, sample_list, is_interp, looped = Image2Video(resolution="320_512"), i2v_examples_loop_512, False, False
    # i2v, sample_list, is_interp, looped = Image2Video(resolution="256_256"), i2v_examples_256, False, False

    for params in sample_list:
        if is_interp and not looped:
            im_path, prompt, steps, cfg_scale, eta, fs, seed, att_start, att_end, stereo_scale, max_warp_step, scale_disparity_factor, refine_non_mask, propagation, dissolve_step, im_path2 = params
        else:
            im_path, prompt, steps, cfg_scale, eta, fs, seed, att_start, att_end, stereo_scale, max_warp_step, scale_disparity_factor, refine_non_mask, propagation, dissolve_step = params
            im_path2 = None
        scales = [stereo_scale / 2, - stereo_scale / 2] if run_both else [stereo_scale]
        for stereo_scale in scales:
            video_path = i2v.get_image(
                np.array(Image.open(im_path).convert("RGB")),
                prompt,
                steps=steps,
                cfg_scale=cfg_scale,
                eta=eta,
                fs=fs,
                seed=seed,
                stereo_scale_factor=stereo_scale,
                is_stereo=True,
                max_warp_step=max_warp_step,
                image2=np.array(Image.open(im_path2).convert("RGB")) if im_path2 else None,
                refine_non_mask=refine_non_mask,
                propagation=propagation,
                looped=looped,
                dissolve_step=dissolve_step,
            )
            print('done', video_path)
