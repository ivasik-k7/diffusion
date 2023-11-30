import argparse
import os
import boto3
import asyncio
import uuid
from PIL import Image
from io import BytesIO
from botocore.exceptions import NoCredentialsError
from generator import DiffusionGenerator
from utils import setup_argparse

KB = 1024
MB = 1024 * KB

# S3 Bucket Information
AWS_S3_BUCKET: str = "mlops-generations"
S3_FOLDER_PATH: str = "images"

supported_file_extensions: list = ["jpeg", "jpg", "png"]

# Create an S3 client
s3 = boto3.client("s3")


async def upload_file(image: Image):
    file_extension: str = "png"
    image_bytes_io = BytesIO()
    image.save(image_bytes_io, format="PNG")

    image_size_bytes = image_bytes_io.getbuffer().nbytes

    if not 0 < image_size_bytes <= 2 * MB:
        print("The file are too big to be uploaded to S3 Bucket")
        return

    try:
        file_extension = image.filename.split(".")[-1]
    except Exception as e:
        file_extension = "png"

    if file_extension not in supported_file_extensions:
        print("The file extenstion not coresponded to expected generations")
        return

    filename: str = "{name}.{extension}".format(
        extension=file_extension,
        name=uuid.uuid4(),
    )

    object_key: str = f"{S3_FOLDER_PATH}/{filename}"

    try:
        image_bytes_io.seek(0)

        s3.upload_fileobj(image_bytes_io, AWS_S3_BUCKET, object_key)

        s3_url = f"https://{AWS_S3_BUCKET}.s3.amazonaws.com/{object_key}"

        print(
            f"The images has been succesfully uploaded with the following url: {s3_url}"
        )
    except NoCredentialsError:
        print("The credentials are not ok!")
    except Exception as e:
        print(f"The error occured during uploading the generations to S3 bucket: ${e}")


async def main():
    args = setup_argparse()

    high_noise_frac = 0.8

    pipeline = DiffusionGenerator.load_diffusion_pipeline(
        model_id=args.model_id,
        use_safetensors=True,
    )

    while True:
        await asyncio.sleep(10)

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
            for i in images:
                await upload_file(i)


if __name__ == "__main__":
    asyncio.run(main())
