import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from authtoken import auth_token

def stable_diffusion(prompt):
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"

    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)

    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=auth_token)
    pipe = pipe.to(device)

    with autocast("cuda"):
        image = pipe(prompt, guidance_scale=8.5).images[0]

    image.save(f"exported-art/{prompt}.png")