from io import BytesIO
from fastapi import FastAPI, Depends, Form
from fastapi.responses import Response, StreamingResponse
import uvicorn
import argparse
import base64
from time import time

from omegaconf import OmegaConf
import base64
import threading
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
from diffusers import StableDiffusionXLPipeline, LCMScheduler
import torch
from pydantic import BaseModel
from io import BytesIO
from huggingface_hub import hf_hub_download
import os
from safetensors.torch import load_file



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10006)
    # parser.add_argument("--config", default="configs/image.yaml")
    return parser.parse_args()


base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "ByteDance/Hyper-SD"
ckpt_name = "Hyper-SDXL-8steps-CFG-lora.safetensors"


class SampleInput(BaseModel):
    prompt: str


class DiffUsers:
    def __init__(self):

        print("setting up model")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16)
        self.pipeline.to(device="cuda", dtype=torch.bfloat16)

        self.unet_state = load_file(hf_hub_download("ByteDance/Hyper-SD", "Hyper-SDXL-1step-Unet.safetensors"), device="cuda")
        self.pipeline.unet.load_state_dict(self.unet_state)
        self.pipeline.scheduler = LCMScheduler.from_config(self.pipeline.scheduler.config, timestep_spacing ="trailing")



        self.steps = 1
        self.guidance_scale = 0
        self._lock = threading.Lock()
        print("model setup done")

    def generate_image(self, prompt: str):
        generator = torch.Generator(self.device)
        seed = generator.seed()
        generator = generator.manual_seed(seed)
        image = self.pipeline(
            prompt="3D icon of a " +prompt.strip(),
            # negative_prompt="lowres, bad anatomy, bad hands, signature, watermarks, ugly, imperfect eyes, skewed eyes, unnatural face, unnatural body, error, extra limb, missing limbs, painting by bad-artist, ugly, blurry, pixelated obscure, unnatural colors, poor lighting, dull, unclear, cropped, lowres, low quality, artifacts, duplicate",
            num_inference_steps=self.steps,
            generator=generator,
            guidance_scale=self.guidance_scale,
            width=1024,
            height=1024,
        ).images[0]

        # Generate a unique file name
        # file_name = f"generated_image_{seed}.png"

        # # Save in a directory named 'generated_images'
        # file_path = os.path.join("generated_images", file_name)
        # # Get the absolute file path
        # file_path = os.path.abspath(file_path)

        # # Ensure the directory exists
        # os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # # Save the image to a file
        # image.save(file_path, format="png")

        # Return the file path
        return image

    def sample(self, input: SampleInput):
        try:
            with self._lock:
                return self.generate_image(input.prompt)
        except Exception as e:
            print(e)
            with self._lock:
                return self.generate_image(input.prompt)