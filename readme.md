# Image Generation using Stable Diffusion Models

This repository contains a Python script for generating images using Stable Diffusion Models. The script leverages the `diffusers` library and provides a command-line interface for easy use.

## Getting Started

### Prerequisites

Make sure you have Python installed on your machine. You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Usage

The script can be run with the following command:

```bash
python main.py -p "Your custom prompt" -m "model_id" -ni 5 -ns 10
```

- **-p, --prompt**: Specify the prompt for generating images (default: "DVD still from 1981 dark fantasy film Excalibur, Frozen Church, red brown toy poodle warrior, dark light, sunshine, portrait").
- **-m, --model_id**: Specify the model ID (default: "stabilityai/stable-diffusion-2-1").

- **-ni, --num_images**: Number of images to generate (default: 10).

- **-ns, --num_interfaces**: Number of interface steps to proceed on the image (Default: 10).

### Script Structure

- **setup_argparse**: Configures command-line arguments for the script.

- **load_diffusion_pipeline**: Loads the Stable Diffusion Pipeline based on the specified model ID.

- **load_diffusion_refiner**: Loads a Diffusion Refiner using a separate model ID, text encoder, and VAE.

- **generate_images**: Generates images based on the provided prompt and pipeline settings.

- **save_images**: Saves the generated images in an output folder.

- **main**: The main function that orchestrates the entire image generation process.

### Configuration

The script uses the `diffusers` library for Stable Diffusion Models. The model and pipeline configurations are specified in the script and can be adjusted as needed.

### Output

Generated images are saved in the "out" folder. Each run creates a new folder with a timestamp to store the images.

## Example Images

![Example 1](examples/dvdstifrodar_0.png)
![Example 2](examples/dvdstifrodar_1.png)
![Example 3](examples/dvdstifrodar_2.png)
