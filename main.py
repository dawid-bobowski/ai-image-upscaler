import torch
from PIL import Image
from RealESRGAN import RealESRGAN
import os, sys
from config import (
    GPU_DEVICE,
    CPU_DEVICE,
    SUPPORTED_UPSCALE_MULTIPLIERS,
    IGNORE_FILES,
    SUPPORTED_EXTENSIONS,
)
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available. Disabling")

# load a device
print("Loading device... ", end="")
device_type = GPU_DEVICE if torch.cuda.is_available() else CPU_DEVICE
print(f"Selected model type is: {device_type}.")
print(f"Starting device... ", end="")
device = torch.device(device_type)
if device_type == GPU_DEVICE:
    tensor = torch.tensor([1, 2, 3])
    tensor = tensor.to(device)
print("Device is ready!\n")

# set up an upscale multiplier
multiplier = 2
cli_multiplier = None if len(sys.argv) < 4 else sys.argv[3]
if cli_multiplier is None:
    print(f"Using default upscale multiplier x{multiplier}.")
elif cli_multiplier not in SUPPORTED_UPSCALE_MULTIPLIERS:
    print(f"Upscale multiplier x{cli_multiplier} is not supported. Using default one set to x{multiplier}.")
else:
    multiplier = cli_multiplier
    print(f"Upscale multiplier set to x{multiplier}.")

# print directories
input_dir = sys.argv[1]
output_dir = sys.argv[2]
print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}\n")

# load an ai model
print("Loading AI model... ", end="")
model = RealESRGAN(device, scale=int(multiplier))
model.load_weights(f"weights/RealESRGAN_x{multiplier}.pth", download=True)
print("Model loaded!")

# upscale images
print("Starting upscaling process...\n")

for file in os.listdir(input_dir):
    filename = os.fsdecode(file)

    if filename in IGNORE_FILES:
        continue

    extension = filename.split(".")[-1]
    upscaled_filename = f"{output_dir}/{filename.split('.')[0]}-x{multiplier}.{extension}"

    if os.path.isfile(upscaled_filename):
        print(f"{filename}: file already upscaled.")
        continue

    if extension in SUPPORTED_EXTENSIONS: 
        print(f"{filename}: file OK.\nProcessing... ", end="")
        image = Image.open(f"{input_dir}\\{filename}").convert("RGB")
        sr_image = model.predict(image)
        sr_image.save(upscaled_filename)
        print("Image saved!")
        continue

    print (f"{filename}: file extension not supported.")

print("\nAll files were processed. Thank you for using AI Image Upscaler!")
