import torch
from PIL import Image
from RealESRGAN import RealESRGAN
import os

GPU_DEVICE = "cuda"
CPU_DEVICE = "cpu"
DIR_IGNORE_LIST = [".gitignore", "upscaled"]
SUPPORTED_EXTENSIONS = ["png", "jpg"]
SUPPORTED_UPSCALE_MULTIPLIERS = ["2", "4", "8"]
IAMGES_PATH = "images/"


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
  if multiplier not in [SUPPORTED_UPSCALE_MULTIPLIERS, ""]:
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
directory = os.fsencode(IAMGES_PATH)

for file in os.listdir(directory):
    filename = os.fsdecode(file)

    if filename in DIR_IGNORE_LIST:
        continue

    extension = filename.split(".")[-1]
    upscaled_filename = f"{IAMGES_PATH}upscaled/{filename.split('.')[0]}-upscaled.{extension}"

    if os.path.isfile(upscaled_filename):
        print(f"{filename}: file already upscaled.")
        continue

    if extension in SUPPORTED_EXTENSIONS: 
        print(f"{filename}: file OK.\nProcessing... ", end="")
        image = Image.open(f"{IAMGES_PATH}{filename}").convert("RGB")
        sr_image = model.predict(image)
        sr_image.save(upscaled_filename)
        print("Image saved!")
        continue

    print (f"{filename}: file extension not supported.")

print("\nAll files were processed. Exitting. Thank you for using AI Image Upscaler!")
