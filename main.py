import torch
from PIL import Image
from RealESRGAN import RealESRGAN
import os

GPU_DEVICE = "cuda"
CPU_DEVICE = "cpu"

supported_extensions = ["png", "jpg"]
supported_upscale_multipliers = ["2", "4", "8"]
images_path = "images/"

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
while True:
  multiplier = input("How much would you like to upscale your images? Available options are: 2, 4 or 8 (press Enter for default=4): ")
  if multiplier not in [supported_upscale_multipliers, ""]:
      print("Your choice has to be one of 2, 4 (default) or 8.")
      continue
  break
if multiplier == "":
    multiplier = "4"
print(f"Chosen upscale multiplier is x{multiplier}.\n")

# load an ai model
print("Loading AI model... ", end="")
model = RealESRGAN(device, scale=int(multiplier))
model.load_weights(f"weights/RealESRGAN_x{multiplier}.pth", download=True)
print("Model loaded!")

# upscale images
print("Starting upscaling process...\n")
directory = os.fsencode(images_path)
for file in os.listdir(directory):
    file_name = os.fsdecode(file)
    extension = file_name.split(".")[-1]
    if extension in supported_extensions: 
        print(f"{file_name}: file OK.\nProcessing... ", end="")
        image = Image.open(f"{images_path}{file_name}").convert("RGB")
        sr_image = model.predict(image)
        sr_image.save(f"{images_path}upscaled/{file_name.split('.')[-2]}-upscaled.{extension}")
        print("Image saved!")
    if file_name == ".gitignore":
        continue
    else:
        print (f"{file_name}: file extension not supported.")
print("\nAll files were processed. Exitting. Thank you for using AI Image Upscaler!")
