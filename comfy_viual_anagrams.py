import folder_paths
from PIL import Image
import numpy as np
import torch
import os

from pathlib import Path

import torch
from diffusers import DiffusionPipeline

from .visual_anagrams.views import get_views
from .visual_anagrams.samplers import sample_stage_1, sample_stage_2
from .visual_anagrams.utils import save_illusion

class VisualAnagramsSampleNode:
    ana_views = ["flip", "rotate_cw", "rotate_180", "negate", "skew", "patch_permute", "pixel_permute", "jigsaw", "inner_circle",]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": ("STRING",{"multiline": True, "default":"use | to split two prompts"}),
                "view":  (cls.ana_views,),
                "steps": ("INT", {"default": 30}),
                "guidance_scale":("FLOAT",{"default": 10.0}),
                "seed": ("INT", {"default": 0, "min":0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "ana_generate"
    CATEGORY = "visial_anagrams"

    def ana_generate(self, prompts, view, steps, guidance_scale, seed):
        # Make models
        stage_1 = DiffusionPipeline.from_pretrained(
                        "DeepFloyd/IF-I-M-v1.0",
                        variant="fp16",
                        torch_dtype=torch.float16)
        stage_2 = DiffusionPipeline.from_pretrained(
                        "DeepFloyd/IF-II-M-v1.0",
                        text_encoder=None,
                        variant="fp16",
                        torch_dtype=torch.float16,
                    )
        stage_1 = stage_1.to("cuda")
        stage_2 = stage_2.to("cuda")

        # Get prompt embeddings
        prompt_embeds = [stage_1.encode_prompt(f'{p}'.strip()) for p in prompts.split("|")]
        prompt_embeds, negative_prompt_embeds = zip(*prompt_embeds)
        prompt_embeds = torch.cat(prompt_embeds)
        negative_prompt_embeds = torch.cat(negative_prompt_embeds)  # These are just null embeds

        # Get views
        views = get_views(['identity', view])

        # Save metadata
        # save_metadata(views, args, save_dir)


        generator = torch.manual_seed(seed)

        # Sample 64x64 image
        image = sample_stage_1(stage_1, 
                                prompt_embeds,
                                negative_prompt_embeds,
                                views,
                                num_inference_steps=steps,
                                guidance_scale=guidance_scale,
                                reduction='mean',
                                generator=generator)

        # Sample 256x256 image, by upsampling 64x64 image
        image = sample_stage_2(stage_2,
                                image,
                                prompt_embeds,
                                negative_prompt_embeds, 
                                views,
                                num_inference_steps=steps,
                                guidance_scale=guidance_scale,
                                reduction='mean',
                                noise_level=50,
                                generator=generator)
        
        # 1 3 256 256 =>1 3 256 256 => 256, 256, 3

        im_views = torch.stack([view.view(image[0]) for view in views])
        print(im_views)
        im_views = im_views / 2. + 0.5
        print(im_views)
        im_views = im_views.squeeze(0).permute(1,2,0)
        return (im_views, )

class VisualAnagramsAnimateNode:
    ...

NODE_CLASS_MAPPINGS = {
    "VisualAnagramsSample": VisualAnagramsSampleNode,
    "VisualAnagramsAnimate": VisualAnagramsAnimateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VisialAnagramsSample": "visual_anagrams_sample",
    "VisualAnagramsAnimate": "visual_anagrams_animate",
}