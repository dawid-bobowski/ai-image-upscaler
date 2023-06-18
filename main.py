import torch
from PIL import Image
from RealESRGAN import RealESRGAN
import os

supported_extensions = ["png", "jpg"]
images_path = "images/"
results_path = "results/"

print("Loading AI model...")
device = torch.device("cpu")
model = RealESRGAN(device, scale=4)
model.load_weights("weights/RealESRGAN_x4.pth", download=True)
print("Model loaded.")

directory = os.fsencode(images_path)
for file in os.listdir(directory):
    file_name = os.fsdecode(file)
    extension = file_name.split(".")[-1]
    if extension in supported_extensions: 
        print(f"{file_name}: file OK.")
        image = Image.open(f"{images_path}{file_name}").convert("RGB")
        print("Starting upscaling process...")
        sr_image = model.predict(image)
        sr_image.save(f"{results_path}{file_name}")
        print("Image saved!")
    else:
        print (f"{file_name}: file extension not supported.")
    print("All files were processed. Exitting. Thank you!")
