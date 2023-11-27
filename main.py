import argparse
import torch
import os
from datetime import datetime
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    DiffusionPipeline,
)


def setup_argparse():
    parser = argparse.ArgumentParser(description="Your script description here.")

    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="DVD still from 1981 dark fantasy film Excalibur, Frozen Church, red brown toy poodle warrior, dark light, sunshine, portrait",
        help="Specify the prompt for generating images (default: cute red brown toy poodle in art brut style)",
    )

    parser.add_argument(
        "-m",
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        help="Specify the model ID (default: stabilityai/stable-diffusion-2-1)",
    )

    parser.add_argument(
        "-ni",
        "--num_images",
        type=int,
        default=1,
        help="Number of images to generate (default: 1)",
    )

    parser.add_argument(
        "-ns",
        "--num_interfaces",
        type=int,
        default=10,
        help="Number of interface steps to proceed on the image (Default: 10)",
    )

    return parser.parse_args()


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


def save_images(
    images: list,
    folder_path: str,
    prompt: str,
):
    os.makedirs(folder_path, exist_ok=True)

    words = prompt.split()
    short_name = "".join(word[:3] for word in words if word.isalpha())[:12].lower()

    for i, image in enumerate(images):
        image_name = f"{short_name}_{i}.png"
        image.save(os.path.join(folder_path, image_name))


def main():
    args = setup_argparse()

    high_noise_frac = 0.8

    pipeline = load_diffusion_pipeline(args.model_id, use_safetensors=True)

    # pipeline.unet = torch.compile(
    #     pipeline.unet,
    #     mode="reduce-overhead",
    #     fullgraph=True,
    # )

    images = generate_images(
        pipeline=pipeline,
        prompt=args.prompt,
        high_noise_frac=high_noise_frac,
        num_inference_steps=args.num_interfaces,
        num_images_per_prompt=args.num_images,
    )

    if images:
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        save_images(
            images=images,
            prompt=args.prompt,
            folder_path=f"out/{current_time}",
        )


if __name__ == "__main__":
    main()
