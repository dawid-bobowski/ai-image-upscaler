# ai-image-upscaler

AI Image Upscaler is a simple Python CLI program which upscales images with JPG and PNG extensions using your GPU (or CPU if no CUDA GPU is available). Currently it allows to use x2, x4 or x8 upscale algorithms.

## To use a GPU mode:

- before doing anything, firstly install CUDA Toolkit 11.7 from Nvidia (the only tested version) available here: https://developer.nvidia.com/cuda-11-7-0-download-archive

## Before you run the program:

- go to ai-image-upscaler main directory;
- run `py -m pip install -r requirements.txt` (it will take a few minutes to install depedning on your Internet connection speed);
- put your images inside of **_images_** folder;
- run: `py main.py`;
- choose upscale multiplier from x2, x4 or x8;
- wait for the results.

Your upscaled images are going to be located inside of **_images/upscaled_** folder with their names appended with "-upscaled".
