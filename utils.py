from PIL import Image
import os
import argparse


def find_output_images(root_folder):
    images = []
    for foldername, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(foldername, filename)
                img = Image.open(file_path)
                images.append(img)

    return images


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


def setup_argparse():
    parser = argparse.ArgumentParser(description="Your script description here.")
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="dvd screenshot of 1987 Dark Souls fantasy film,abstract lonely red brown toy poodle as a necromancer in the dungeon",
        help="Specify the prompt for generating images",
    )

    parser.add_argument(
        "-np",
        "--negative_prompt",
        type=str,
        default=", ".join(
            [
                "bad anatomy",
                "bad proportions",
                "blurry",
                "cloned face",
                "cropped",
                "deformed",
                "dehydrated",
                "disfigured",
                "duplicate",
                "error",
                "extra arms",
                "extra fingers",
                "extra legs",
                "extra limbs",
                "fused fingers",
                "gross proportions",
                "jpeg artifacts",
                "long neck",
                "low quality",
                "lowres",
                "malformed limbs",
                "missing arms",
                "missing legs",
                "morbid",
                "mutated hands",
                "mutation",
                "mutilated",
                "out of frame",
                "poorly drawn face",
                "poorly drawn hands",
                "signature",
                "text",
                "too many fingers",
                "ugly",
                "username",
                "watermark",
                "worst quality",
            ]
        ),
        help="Specify the negative prompt for generating images",
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
        default=2,
        help="Number of images to generate (default: 1)",
    )

    parser.add_argument(
        "-ns",
        "--num_interfaces",
        type=int,
        default=25,
        help="Number of interface steps to proceed on the image (default: 25)",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Width of the generated image (default: 768)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Height of the generated image (default: 768)",
    )

    return parser.parse_args()
