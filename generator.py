import torch
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    DiffusionPipeline,
)


class DiffusionGenerator:
    @staticmethod
    def generate_images(
        *,
        pipeline,
        prompt: str,
        high_noise_frac: float,
        num_inference_steps: int,
        num_images_per_prompt: int,
    ):
        result = pipeline(
            prompt=prompt,
            denoising_end=high_noise_frac,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        return result

    @staticmethod
    def load_diffusion_refiner(
        refiner_model_id,
        text_encoder,
        vae,
        use_safetensors=True,
    ):
        refiner = DiffusionPipeline.from_pretrained(
            refiner_model_id,
            text_encoder_2=text_encoder,
            vae=vae,
            torch_dtype=torch.float32,
            use_safetensors=use_safetensors,
        )

        return refiner

    @staticmethod
    def load_diffusion_pipeline(
        model_id,
        use_safetensors=True,
    ):
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            use_safetensors=use_safetensors,
        )

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

        return pipe
