import os
from generator import DiffusionGenerator
from datetime import datetime
from utils import setup_argparse, save_images


def main():
    args = setup_argparse()

    high_noise_frac = 0.8

    pipeline = DiffusionGenerator.load_diffusion_pipeline(
        args.model_id,
        use_safetensors=True,
    )

    images = DiffusionGenerator.generate_images(
        pipeline=pipeline,
        prompt=args.prompt,
        num_inference_steps=args.num_interfaces,
        num_images_per_prompt=args.num_images,
        negative_prompt=args.negative_prompt,
        high_noise_frac=high_noise_frac,
        height=args.height,
        width=args.width,
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
