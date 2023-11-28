import argparse
import boto3
import asyncio
import uuid
from botocore.exceptions import NoCredentialsError
from generator import DiffusionGenerator

KB = 1024
MB = 1024 * KB

# S3 Bucket Information
AWS_S3_BUCKET: str = "mlops-generations"
S3_FOLDER_PATH: str = "images"

supported_file_extensions: list = ["jpeg", "jpg", "png"]

# Create an S3 client
s3 = boto3.client("s3")


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


async def upload_file(image):
    file_content = await image.read()

    file_size = len(file_content)

    if not 0 < file_size <= 2 * MB:
        print("The file are too big to be uploaded to S3 Bucket")
        return

    file_extension: str = image.filename.split(".")[-1]

    filename: str = "{name}.{extension}".format(
        extension=file_extension,
        name=uuid.uuid4(),
    )

    object_key: str = f"{S3_FOLDER_PATH}/{filename}"

    if file_extension not in supported_file_extensions:
        print("The file extenstion not coresponded to expected generations")
        return

    try:
        image.file.seek(0)

        s3.upload_fileobj(image.file, AWS_S3_BUCKET, object_key)

        s3_url = f"https://{AWS_S3_BUCKET}.s3.amazonaws.com/{object_key}"
        print(
            f"The images has been succesfully uploaded with the following url: {s3_url}"
        )
    except NoCredentialsError:
        print("The credentials are not ok!")
    except Exception as e:
        print(f"The error occured during uploading the generations to S3 bucket: ${e}")


async def main():
    arguments = setup_argparse()

    high_noise_frac = 0.8

    pipeline = DiffusionGenerator.load_diffusion_pipeline(
        model_id=arguments.model_id,
        use_safetensors=True,
    )

    while True:
        await asyncio.sleep(10)

        images = DiffusionGenerator.generate_images(
            pipeline=pipeline,
            prompt=arguments.prompt,
            high_noise_frac=high_noise_frac,
            num_inference_steps=arguments.num_interfaces,
            num_images_per_prompt=arguments.num_images,
        )

        if images:
            for i in images:
                upload_file(i)


if __name__ == "__main__":
    asyncio.run(main())
