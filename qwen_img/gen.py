from diffusers import DiffusionPipeline # pyright: ignore[reportPrivateImportUsage]
import torch

model_name = "Qwen/Qwen-Image"

# Load the pipeline
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "mps"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)

# Generate image
prompt = '''
Two zombies in tracksuits sitting in the rusty old soviet car.
'''

negative_prompt = " "


# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "640x480" : (640, 480),
    "320x240" : (320, 240),
}

width, height = aspect_ratios["320x240"]

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="mps").manual_seed(42)
).images[0] # type: ignore

image.save("example.png")

