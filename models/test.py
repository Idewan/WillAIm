from diffusers import StableDiffusionPipeline
import torch

rmodel_id = "~/home/ild2105/WillAIm/models/stable-diffusion-v1-5"
model_id = "./stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cuda")

prompt = "anime cat father running around a supermarket buying groceries"
image = pipe(prompt).images[0]

image.save("test.png")
