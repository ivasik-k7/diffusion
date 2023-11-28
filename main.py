import os
import argparse
from generator import DiffusionGenerator
from datetime import datetime


def setup_argparse():
    parser = argparse.ArgumentParser(description="Your script description here.")

    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="DVD still from Dark Fantasy Film The Legend 1985,, red brown toy poodle as a hedge knight in the dungeon",
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
        default=25,
        help="Number of interface steps to proceed on the image (default: 25)",
    )

    return parser.parse_args()


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

    pipeline = DiffusionGenerator.load_diffusion_pipeline(
        args.model_id,
        use_safetensors=True,
    )

    images = DiffusionGenerator.generate_images(
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
