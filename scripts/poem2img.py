from diffusers import StableDiffusionPipeline
from torch import autocast
from tqdm import tqdm

import torch
import random
import json


class Model():

    def __init__(self, model_id):
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id,
            revision="fp16",
            torch_dtype=torch.float16, 
            use_auth_token=True
        ).to("cuda")

class Prompt():

    def __init__(self):
        self.prompt_supplement = [
            "An artistic and dream-like represenation of how it feels for",
            "An abstract painting of the symbols in",
            "A realistic painting by a dutch master of the symbols",
            "An impressionist painting of the symbols in",
            "An impressionist painting of the metaphors in ",
            "A painting of emotions and metaphors evoked in",
            ""
            ]
        self.num_prompt_supp = len(self.prompt_supplement)

    def clean_prompt(self, prompt):
        """

        """
        prompt = prompt.replace('\r', '')
        prompt = prompt.replace('\n', '')

        return prompt
    
    def get_prompts(self, poem):
        """
        
        """
        supplement_1_ind = random.randrange(0, self.num_prompt_supp)
        supplement_2_ind = random.randrange(0, self.num_prompt_supp)

        prompt_1_supp = self.prompt_supplement[supplement_1_ind]
        prompt_2_supp = self.prompt_supplement[supplement_2_ind]
        
        prompt_1 = prompt_1_supp + "\"" + poem + "\""
        prompt_2 = prompt_2_supp + "\"" + poem + "\""

        return [prompt_1, prompt_2]

if __name__ == '__main__':
    # Store image location, prompt and poem together
    data = {
        "info": "Stores image location, prompt used to generate, and poem together",
        "poem2img": []
        }

    model_id = "CompVis/stable-diffusion-v1-4"

    ldm_model = Model(model_id)
    prompt_gen = Prompt()

    f1 = open("data/poem/poem.json")
    data_poem = json.load(f1)["poems"]
    f1.close()

    for sub_data in tqdm(data_poem):
        temp_poem = sub_data['poem']
        c_poem = temp_poem.encode("ascii", "ignore")
        c_poem = temp_poem.decode()

        poem = prompt_gen.clean_prompt(c_poem)
        prompts = prompt_gen.get_prompts(poem)

        for i in range(2):

            with autocast("cuda"):
                image = ldm_model.pipe(prompts[i], guidance_scale=7.5).images[0]

            img_path = f"data/image/{sub_data['ID']}/{i}.png"
            image.save(img_path)

            data["poem2img"].append({
                "id":sub_data['ID'],
                "poem":c_poem,
                "prompt":prompts[i],
                "img_path":img_path,
                "keywords":sub_data['keywords']
            })
    
    with open("data/poem2img.json", "w") as f:
        json.dump(data, f)
    
    print("Done!")
